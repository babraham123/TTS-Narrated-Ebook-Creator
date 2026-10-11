"""Command line entry point: calliope IN.epub OUT.epub --voice WAV --lang xx."""

import argparse
import os
import shutil
import signal
import sys

from calliope.pipeline import Options, run
from calliope.qa import QAConfig, WhisperTranscriber
from calliope.tts import MODELS, ChatterboxBackend, FakeBackend, resolve_model

EXIT_FAILURE = 1
EXIT_SIGINT = 130
EXIT_SIGTERM = 143

# Test hook: comma-separated sentence indices (0-based, book order) whose clips are replaced by
# silence, to exercise the retry path. "IDX:N" silences only the first N attempts.
FORCE_SILENCE_ENV = "CALLIOPE_FORCE_SILENCE"


class Terminated(BaseException):
    """Raised from the SIGTERM handler so cleanup (temp file removal) runs on the way out."""


def _raise_terminated(signum, frame):
    signal.signal(signal.SIGTERM, signal.SIG_DFL)  # a second SIGTERM kills immediately
    raise Terminated()


def default_work_dir(output):
    out_dir = os.path.dirname(os.path.abspath(output))
    stem = os.path.splitext(os.path.basename(output))[0]
    return os.path.join(out_dir, f".{stem}.calliope-work")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="calliope",
        description="Convert an EPUB into an EPUB 3 with Media Overlays narrated by Chatterbox TTS.",
    )
    parser.add_argument("input", help="input .epub")
    parser.add_argument("output", help="output .epub")
    parser.add_argument("--voice", "-v", required=True, help="reference voice sample (WAV, longer than 5 s)")
    parser.add_argument("--lang", "--language", "-l", dest="lang", default="en",
                        help="book language code (default: en)")
    parser.add_argument("--model", choices=MODELS, default=None,
                        help="Chatterbox model; default: turbo for --lang en, multilingual otherwise "
                             "(turbo is English-only)")
    parser.add_argument("--work-dir", default=None,
                        help="sentence cache and scratch space; rerunning with the same work dir resumes "
                             "(default: .<OUT stem>.calliope-work next to OUT)")
    parser.add_argument("--device", default="auto", help="torch device: auto (default), cuda, cpu")
    parser.add_argument("--seed", type=int, default=0, help="base random seed (default: 0)")
    parser.add_argument("--skip-audio", action="store_true",
                        help="debug layout without TTS: narrate with a placeholder tone, no model load")

    qa = parser.add_argument_group("quality checks")
    qa.add_argument("--max-retries", type=int, default=2,
                    help="extra attempts (new seed each) for a clip that fails the checks (default: 2)")
    qa.add_argument("--chars-per-second", type=float, default=None,
                    help="speaking rate used to predict clip length (default: 14; zh 4.5, ja/ko 7)")
    qa.add_argument("--max-silence", type=float, default=None,
                    help="reject clips whose share of silent frames exceeds this (default: 0.6)")
    qa.add_argument("--whisper-check", action="store_true",
                    help="transcribe every clip with faster-whisper and flag high error rates")
    qa.add_argument("--whisper-model", default="small", help="faster-whisper model size (default: small)")
    qa.add_argument("--whisper-compute-type", default="int8",
                    help="CTranslate2 compute type (default: int8, which Pascal GPUs support)")
    qa.add_argument("--wer-threshold", type=float, default=0.3,
                    help="flag clips whose word error rate exceeds this (default: 0.3)")
    return parser


def parse_force_silence(value):
    forced = {}
    for part in filter(None, (p.strip() for p in (value or "").split(","))):
        index, _, count = part.partition(":")
        forced[int(index)] = int(count) if count else None
    return forced


def parse_arguments(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.model = resolve_model(args.model, args.lang)
    except ValueError as e:
        parser.error(str(e))
    if not os.path.isfile(args.input):
        parser.error(f"input not found: {args.input}")
    if not os.path.isfile(args.voice):
        parser.error(f"voice sample not found: {args.voice}")
    if os.path.abspath(args.input) == os.path.abspath(args.output):
        parser.error("output must differ from input")
    if args.max_retries < 0:
        parser.error("--max-retries must be >= 0")
    try:
        args.force_silence = parse_force_silence(os.environ.get(FORCE_SILENCE_ENV))
    except ValueError:
        parser.error(f"{FORCE_SILENCE_ENV} must look like '3,7:1'")
    args.work_dir = args.work_dir or default_work_dir(args.output)
    return args


def make_backend_factory(args):
    if args.skip_audio:
        return lambda: FakeBackend(delay=float(os.environ.get("CALLIOPE_FAKE_DELAY", "0")))

    def load():
        print(f"Loading Chatterbox {args.model} on {args.device}...")
        return ChatterboxBackend(args.model, args.lang, args.voice, args.device)

    return load


class WhisperFactory:
    def __init__(self, args):
        self.args = args
        self.name = f"faster-whisper:{args.whisper_model}:{args.whisper_compute_type}"

    def __call__(self):
        print(f"Loading faster-whisper {self.args.whisper_model}...")
        device = "auto" if self.args.device == "auto" else self.args.device.split(":")[0]
        return WhisperTranscriber(self.args.whisper_model, self.args.lang, device, self.args.whisper_compute_type)


def main(argv=None):
    args = parse_arguments(argv)
    signal.signal(signal.SIGTERM, _raise_terminated)

    opts = Options(
        input=args.input,
        output=args.output,
        voice=args.voice,
        lang=args.lang,
        model="fake" if args.skip_audio else args.model,
        work_dir=args.work_dir,
        seed=args.seed,
        max_retries=args.max_retries,
        qa=QAConfig.for_language(args.lang, chars_per_second=args.chars_per_second,
                                 max_silence=args.max_silence),
        wer_threshold=args.wer_threshold,
        force_silence=args.force_silence,
    )
    try:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH")
        whisper = WhisperFactory(args) if args.whisper_check else None
        stats = run(opts, make_backend_factory(args), whisper)
    except Terminated:
        print("calliope: terminated; finished sentences are cached in the work dir", file=sys.stderr)
        return EXIT_SIGTERM
    except KeyboardInterrupt:
        print("calliope: interrupted; finished sentences are cached in the work dir", file=sys.stderr)
        return EXIT_SIGINT
    except Exception as e:
        print(f"calliope: error: {e}", file=sys.stderr)
        return EXIT_FAILURE

    print(f"Wrote {args.output}: {stats['sentences']} sentences, {stats['synthesized']} synthesized, "
          f"{stats['cached']} cached, {len(stats['flagged'])} flagged")
    return 0

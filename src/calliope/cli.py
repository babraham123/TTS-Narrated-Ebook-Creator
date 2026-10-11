"""Command line entry point: calliope IN.epub OUT.epub --voice WAV --lang xx."""

import argparse
import os
import shutil
import signal
import sys

from calliope.pipeline import Options, run
from calliope.tts import MODELS, ChatterboxBackend, FakeBackend, resolve_model

EXIT_FAILURE = 1
EXIT_SIGINT = 130
EXIT_SIGTERM = 143


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
    return parser


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
    args.work_dir = args.work_dir or default_work_dir(args.output)
    return args


def make_backend_factory(args):
    if args.skip_audio:
        return lambda: FakeBackend(delay=float(os.environ.get("CALLIOPE_FAKE_DELAY", "0")))

    def load():
        print(f"Loading Chatterbox {args.model} on {args.device}...")
        return ChatterboxBackend(args.model, args.lang, args.voice, args.device)

    return load


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
    )
    try:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH")
        stats = run(opts, make_backend_factory(args))
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
          f"{stats['cached']} cached")
    return 0

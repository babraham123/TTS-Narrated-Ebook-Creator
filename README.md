# Calliope: EPUB 3 Media Overlays Creator using TTS

**Turn a text e-book into a narrated e-book with text highlighting in the EPUB 3 Media Overlays file format.**

## Key Features
* **🔒 Offline:** The method runs entirely offline, and thereby eliminating the recurring costs, privacy concerns, and copyright compliance issues associated with cloud-based solutions.
* **🎯 Zero Drift:** Calculating timestamps during TTS rather than forced alignment. The audio and text never go out of sync.
* **🎨 Layout Preservation:** Publisher’s original typography, styling, and embedded media are preserved.
* **🌓 Dark Mode Support:** Automatically generates high-contrast highlighting, e.g. purple for dark mode, yellow for light mode.
* 🧠 Supports the TTS methods **Chatterbox** and **XTTS-v2**.
* **Resumable:** every sentence is cached, so an interrupted run picks up where it stopped and produces the same EPUB.
* **Quality checks:** truncated, rambling or silent clips are retried with a new seed. Optionally, Whisper transcribes every clip to catch misread text.

---

## 🎥 See Calliope in Action

Watch a short demonstration of the system in use. This video shows the conversion process and the final playback in the Thorium reader, demonstrating the exact synchronization between the text highlighting and the TTS audio.

[**▶️ Watch the Video Demonstration**](https://youtu.be/j15BHY1hh7w)

---

## Prerequisites
1.  **Python 3.12** (the Chatterbox stack is pinned and locked for 3.12).
2.  **FFmpeg** on your `PATH` (audio is encoded to MP3 with `ffmpeg`).
    * **Ubuntu/Debian:** `sudo apt install ffmpeg`
    * **MacOS:** `brew install ffmpeg`
    * **Windows:** `winget install ffmpeg`
3.  For GPU synthesis: an NVIDIA GPU and driver supporting CUDA 12.4 (see [Running on a Tesla P4](#running-on-a-tesla-p4-pascal-sm_61)).

Or use the [container image](#container), which bundles everything except the model weights.

---

## Installation
The Chatterbox pipeline is a Python package (`src/calliope`) with a `calliope` command.
Dependencies are pinned in `pyproject.toml` and locked in `uv.lock`: `chatterbox-tts==0.1.7`
with `torch==2.6.0` / `torchaudio==2.6.0`, which on Linux x86_64 come from the CUDA 12.4
(`cu124`) PyTorch index.

```bash
uv sync --extra tts                    # add --extra whisper for --whisper-check
uv run calliope --help
```

The lock covers Linux x86_64 (CUDA 12.4 wheels) and Apple Silicon macOS (PyPI wheels, CPU/MPS).

XTTS is unchanged from upstream: `pip install -r requirements/requirements_xtts.txt` in a
separate Python 3.11 environment and run `python src/epub_processor_xtts.py`.

---

## Usage
You need a clean reference voice sample (WAV, ~15 seconds; Chatterbox requires more than 5 s) to clone the narrator's voice. A sample in English is provided in assets/neutral_narrator.wav (Google translate narrator). Alternatively, several samples, for multiple languages, can be found at [Chatterbox TTS Demo Samples](https://resemble-ai.github.io/chatterbox_demopage/).

```bash
calliope my_book.epub my_book_narrated.epub --voice assets/neutral_narrator.wav --lang en
calliope roman.epub roman_lest.epub --voice stemme.wav --lang no --work-dir /tmp/roman-work
```

`python src/epub_processor_chatterbox.py ...` still works and takes the same arguments.

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `IN` | Source EPUB. |
| `OUT` | Narrated EPUB to write. The run report goes to `OUT.report.json` (`OUT` with its extension replaced) in the same directory. |
| `--voice`, `-v` | Required. Reference WAV for voice cloning (longer than 5 s). |
| `--lang`, `--language`, `-l` | Book language code (default `en`). Also selects the sentence splitter. |
| `--model turbo\|multilingual` | `turbo` = `chatterbox.tts_turbo.ChatterboxTurboTTS` (English only). `multilingual` = `chatterbox.mtl_tts.ChatterboxMultilingualTTS` (23 languages). Default: `turbo` for `en`, `multilingual` otherwise. |
| `--work-dir DIR` | Sentence cache and scratch space. Rerunning with the same directory resumes (default `.<OUT stem>.calliope-work` next to `OUT`). |
| `--device` | `auto` (default), `cuda`, `cuda:1`, `cpu`. |
| `--seed N` | Base random seed (default 0). Every attempt's seed is derived from it and the sentence. |
| `--max-retries N` | Extra attempts, each with a new seed, for a clip that fails the quality checks (default 2). |
| `--chars-per-second X` | Speaking rate used to predict clip length (default 14; 4.5 for `zh`, 7 for `ja`/`ko`). |
| `--max-silence F` | Reject clips with more than this share of silent frames (default 0.6). |
| `--whisper-check` | Transcribe each clip with faster-whisper and flag high error rates (needs the `whisper` extra). |
| `--whisper-model`, `--whisper-compute-type`, `--wer-threshold` | faster-whisper model (default `small`), CTranslate2 compute type (default `int8`) and flag threshold (default 0.3). |
| `--skip-audio` | Debug: narrate with a placeholder tone instead of TTS (no model download, no torch needed). |

Exit status: `0` success (even if some clips are flagged; see the report), `1` failure,
`2` usage error, `143` SIGTERM, `130` Ctrl-C.

### Resuming
Each sentence's audio is cached in `--work-dir` under a key that hashes the input book, the
sentence text, the model (and `chatterbox-tts` version), the language, the voice sample and
the generation settings. Clips are written atomically (temp file + rename), so you can kill
the process at any point, SIGTERM included, and rerun the same command. Only missing
sentences are synthesized, and the model is not loaded at all if none are missing. The
EPUB is always rebuilt from the cached clips with deterministic packaging, so an interrupted
and resumed run produces the same bytes as an uninterrupted one. Delete the work directory
to start over.

### Quality checks
Every synthesized sentence is checked before it is accepted:

* **Duration:** under 0.4x the length predicted from the text (truncated), or over
  2.5x + 1 s (rambling or repetition).
* **Silence:** more than `--max-silence` of 20 ms frames below -40 dBFS.

A failing clip is retried with a new seed up to `--max-retries` times. If no attempt passes,
the best one is kept and listed under `flagged` in the report. Rerunning with a larger
`--max-retries` continues from the attempts already made.

With `--whisper-check`, each clip is also transcribed with faster-whisper. Clips whose word
error rate (character error rate for `zh`/`ja`) exceeds `--wer-threshold` are flagged but not
retried. Digits and spelled-out numbers count as different words, so expect some false
positives.

For testing the retry path, `CALLIOPE_FORCE_SILENCE=3,7:1` replaces the clips of the given
0-based sentence indices (book order) with silence: on every attempt, or only on the first
N attempts with `IDX:N`.

### Report
`OUT.report.json` is written on every run, including failed or interrupted ones (`status`:
`ok`, `failed`, `terminated`, `interrupted`). It contains chapter and sentence counts, clips
synthesized and cached, attempts and retries, the flagged sentences with their reasons and
metrics, audio and wall-clock seconds, and:

* `real_time_factor` = `wall_seconds / audio_seconds` for the whole run;
* `synthesis_real_time_factor` = `synthesis_seconds / synthesized_audio_seconds` for the TTS
  work done in this run.

Lower is faster; below 1 is faster than real time.

---

## Container
`Containerfile` builds a CUDA 12.4 image (`nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`) with
Python 3.12, ffmpeg and the locked stack, including faster-whisper. The `calliope` command is
the ENTRYPOINT, and `calliope` and `python3` are both on `PATH`. Model weights are not in the
image: `HF_HOME=/cache`, so mount a volume there and the weights download once. The working
directory is `/books`. Build for `linux/amd64`.

```bash
podman build -t calliope -f Containerfile .

podman run --rm --gpus=all -v calliope-cache:/cache -v ./books:/books calliope \
    IN.epub OUT.epub --voice voice.wav --lang en --work-dir /books/.work
```

`--gpus=all` requires the NVIDIA Container Toolkit. With CDI, use
`--device nvidia.com/gpu=all` instead. Keep `--work-dir` on a mounted volume so an
interrupted container can resume.

### Batch runner
`batch/calliope_batch.py`, in the image as `/opt/calliope_batch.py`, works through an inbox one
book per run: it narrates the oldest EPUB, delivers the narrated EPUB and its report, and posts
the outcome to ntfy. Run it as the container's entrypoint and start it again (a systemd path
unit on the inbox and a timer, say) until the inbox is empty.

```bash
podman run --rm --gpus=all -v calliope-cache:/cache -v ./narrate:/narrate \
    -v ./readalouds:/readalouds -v ./voices:/voices:ro \
    --entrypoint python3 calliope /opt/calliope_batch.py
```

A book is `Title.epub` in `inbox/`, with an optional sidecar `Title.json`:

| Field | Default | |
|-------|---------|---|
| `lang` | `en` | ISO 639 code, passed to `--lang` |
| `voice` | `default` | `<voice>.wav` in the voices directory, passed to `--voice` |
| `collection` | `books` | the directory under the readalouds directory to deliver to |

`voice` and `collection` are letters, digits, `_` and `-` only. A run:

* narrates the book with `work/Title/cache` as its `--work-dir` and delivers
  `Title.readaloud.epub` and `Title.readaloud.report.json` to `<collection>/Title/` under the
  readalouds directory, the report first, so a reader importing the directory never sees the
  EPUB alone;
* on an invalid sidecar or a failed `calliope`, moves the book, its sidecar and any output to
  `failed/` (`failed/Title/` once `calliope` has run);
* on SIGTERM or Ctrl-C, stops `calliope` and leaves the book in the inbox, and the next run
  resumes it from the sentence cache.

It exits 0 when the book is done or the inbox is empty, 1 when the book failed.

| Variable | Default | |
|----------|---------|---|
| `NARRATE_DIR` | `/narrate` | holds `inbox/`, `work/` and `failed/` |
| `READALOUDS_DIR` | `/readalouds` | delivers to `<collection>/Title/` under it |
| `VOICES_DIR` | `/voices` | the reference voice samples |
| `READALOUDS_OWNER` | `0:0` | `UID:GID` the delivered files and their directories belong to |
| `NTFY_URL`, `NTFY_ALERT_ACCESS_TOKEN` | unset | the ntfy topic URL and bearer token; no post without both |

### Running on a Tesla P4 (Pascal, sm_61)
* **Why CUDA 12.4 and torch 2.6.0:** `torch==2.6.0` is the version `chatterbox-tts` 0.1.7 pins,
  and its `cu124` wheels are built for `sm_50;sm_60;sm_70;sm_75;sm_80;sm_86;sm_90`. The
  `sm_60` code runs on the P4's `sm_61`. Newer CUDA 12.8+ PyTorch builds are dropping Pascal,
  and CUDA 13 removes it, so stay on `cu124`.
* **Driver:** CUDA 12.4 needs an R550+ driver, or R525+ via minor-version compatibility.
* **faster-whisper:** pinned to `ctranslate2==4.6.2`, the last CTranslate2 built against
  CUDA 12.4 and cuDNN 9.1. Later releases target CUDA 12.8. The default `int8` compute type
  is supported on Pascal (compute capability 6.1); float16 is not efficient there.
* **Precision and memory:** Chatterbox runs in float32, which is what Pascal is good at.
  Turbo or Multilingual fits in the P4's 8 GB, with room for the Whisper `small` model.
* **Speed:** see `synthesis_real_time_factor` in the report. Pascal has no tensor cores,
  so expect it to be far slower than on recent GPUs.

---

## How It Works
1. **Extraction:** the EPUB is extracted afresh into the work directory, and the OPF and spine are parsed.
2. **Injection:** in every XHTML chapter, text blocks are split into sentences (punkt for the book language). Each sentence is wrapped in a `<span id>` without disturbing the surrounding HTML/CSS.
3. **Synthesis:** the text sent to the model has numbers and `%`, `°`/`°C`/`°F`, `$ € £ ¥`, `&` and `#1` spelled out in the book's language with [num2words](https://pypi.org/project/num2words/) (da, de, en, es, fr, it, nl, no, pt, sv; other languages keep digits and drop the symbols), and other symbols removed. Cardinals and decimals follow the language's separators (`1,250.5` in English, `1.250,5` elsewhere); ordinals (`2nd`, `3.`, `1er`, `2ª`, `3:e`), years (`in 1984`, `1914–1918`), times, Roman numerals after a chapter keyword or a monarch's name, and phone or ISBN numbers (digit by digit) are recognised. Anything ambiguous, such as `3.5.1990` or `v1.2.3`, is sent as written. The XHTML keeps the author's text. Uncached sentences are sent to the local model. Long sentences are split at punctuation, synthesized in chunks and crossfaded. Each clip is checked, retried if needed, and cached.
4. **Alignment:** the chapter audio is the cached clips plus 150 ms gaps, so SMIL clip times come from exact sample counts. Each clip runs until the next one begins, so highlighting is continuous.
5. **Packaging:** one MP3 per chapter, SMIL overlays, `media:duration` (per overlay and total) and `media:active-class` metadata with a matching CSS rule. The EPUB is written reproducibly: `mimetype` first and stored, sorted entries, fixed timestamps.

---

## Development
```bash
uv sync --group dev          # light dependencies only: no torch, no model
uv run python -I -m pytest
```
The tests run the whole pipeline with `--skip-audio`'s fake backend against a generated
EPUB. They cover resume (byte-identical output, nothing re-synthesized), SIGTERM mid-run,
the QA retry path, the report and the package structure. If `epubcheck` is on `PATH`
(`brew install epubcheck`), they also validate the output with it.

---

## Troubleshooting
"FFmpeg not found" Ensure FFmpeg is installed and added to your system PATH. Run ffmpeg -version in your terminal to verify.<br>
"Out of Memory" (CUDA) Neural TTS is heavy. Use `--device cpu` (slower, but works). Synthesis errors are retried like rejected clips. The run fails only if every attempt for a sentence fails, and finished sentences stay cached.<br>
Many flagged clips: check `expected_duration` vs. `duration` in the report. If your voice speaks much faster or slower than average, adjust `--chars-per-second`.

---

## License & Acknowledgements
**Code:** The source code in this repository is released under the **MIT License**. See `LICENSE` for details.

**Models:**
* **Chatterbox:** Uses the MIT license.
* **XTTS-v2:** Released under the **Coqui Public Model License (CPML)**. 


## Citation

If you use this code for your research, please cite the following paper:

> Hammer, H. L., Thambawita, V., & Halvorsen, P. (2026, April). Calliope: A TTS-based Narrated E-book Creator Ensuring Exact Synchronization, Privacy, and Layout Fidelity. In *Proceedings of the ACM Multimedia Systems Conference 2026* (pp. 416-422).

**BibTeX:**
```bibtex
@inproceedings{hammer2026calliope,
  title={Calliope: A TTS-based Narrated E-book Creator Ensuring Exact Synchronization, Privacy, and Layout Fidelity},
  author={Hammer, Hugo L and Thambawita, Vajira and Halvorsen, P{\aa}l},
  booktitle={Proceedings of the ACM Multimedia Systems Conference 2026},
  pages={416--422},
  year={2026}
}
```

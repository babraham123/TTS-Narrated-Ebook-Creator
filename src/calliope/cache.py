"""Work-directory cache of synthesized sentences and encoded chapters.

Each sentence clip is stored as <work>/clips/<k[:2]>/<k>.flac plus <k>.json, where k hashes
everything that determines the audio: the input book, the sentence text, the model, the
language, the voice sample and the generation settings. The JSON is written last, so its
presence marks a complete entry; both files are written atomically, so killing the process at
any point leaves only complete entries (or none) behind.
"""

import hashlib
import io
import json
import os

import soundfile as sf

from calliope.files import atomic_write_bytes, atomic_write_text

CACHE_VERSION = 1


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _digest(payload):
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def clip_key(book_sha, text, model_id, lang, voice_sha, settings):
    return _digest(
        {
            "v": CACHE_VERSION,
            "book": book_sha,
            "text": text,
            "model": model_id,
            "lang": lang,
            "voice": voice_sha,
            "settings": settings,
        }
    )


def attempt_seed(key, attempt):
    """Deterministic 32-bit seed for the nth synthesis attempt of a clip."""
    return int(hashlib.sha256(f"{key}:{attempt}".encode()).hexdigest()[:8], 16)


class ClipCache:
    def __init__(self, work_dir):
        self.clips_dir = os.path.join(work_dir, "clips")
        self.chapters_dir = os.path.join(work_dir, "chapters")

    def _base(self, key):
        return os.path.join(self.clips_dir, key[:2], key)

    def has(self, key):
        return os.path.exists(self._base(key) + ".json")

    def load(self, key):
        """Return (wav, sr, meta) for a complete entry, else None."""
        base = self._base(key)
        try:
            with open(base + ".json", encoding="utf-8") as f:
                meta = json.load(f)
            wav, sr = sf.read(base + ".flac", dtype="float32")
        except (OSError, ValueError, RuntimeError, sf.LibsndfileError):
            return None
        return wav, sr, meta

    @staticmethod
    def encode(wav, sr):
        """Encode a clip the way it is stored. Returns (flac_bytes, wav as it will load back),
        so quality checks see exactly the audio a later run will read from the cache."""
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="FLAC", subtype="PCM_16")
        data = buf.getvalue()
        decoded, _sr = sf.read(io.BytesIO(data), dtype="float32")
        return data, decoded

    def store(self, key, flac_bytes, meta):
        base = self._base(key)
        atomic_write_bytes(base + ".flac", flac_bytes)
        atomic_write_text(
            base + ".json", json.dumps(meta, sort_keys=True, ensure_ascii=False, indent=1)
        )

    def _transcript_path(self, key, transcriber_name):
        return self._base(key) + f".{_digest(transcriber_name)[:12]}.transcript.json"

    def load_transcript(self, key, transcriber_name, attempts):
        """Cached transcript of the clip, if made by this transcriber for this clip version."""
        try:
            with open(self._transcript_path(key, transcriber_name), encoding="utf-8") as f:
                entry = json.load(f)
        except (OSError, ValueError):
            return None
        return entry["text"] if entry.get("attempts") == attempts else None

    def store_transcript(self, key, transcriber_name, attempts, text):
        entry = {"transcriber": transcriber_name, "attempts": attempts, "text": text}
        atomic_write_text(
            self._transcript_path(key, transcriber_name), json.dumps(entry, ensure_ascii=False)
        )

    def chapter_path(self, clip_keys, encoding):
        """Location of the encoded MP3 for a chapter made of these clips, in this order."""
        return os.path.join(
            self.chapters_dir, _digest({"clips": clip_keys, "encoding": encoding}) + ".mp3"
        )

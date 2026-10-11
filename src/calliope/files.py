"""Atomic file writes: write to a temp file in the target directory, then rename over the target.

A reader (or a rerun after a crash or SIGTERM) therefore sees either the old file or the
complete new one, never a partial write.
"""

import contextlib
import os
import tempfile


@contextlib.contextmanager
def atomic_output(path):
    """Yield a temp path next to `path`; on success it is fsynced and renamed to `path`."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix="-" + os.path.basename(path))
    os.close(fd)
    try:
        yield tmp
        with open(tmp, "rb+") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def atomic_write_bytes(path, data):
    with atomic_output(path) as tmp:
        with open(tmp, "wb") as f:
            f.write(data)


def atomic_write_text(path, text):
    atomic_write_bytes(path, text.encode("utf-8"))

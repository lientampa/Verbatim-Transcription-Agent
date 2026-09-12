"""Per-run content identity and shared microsecond offset representation."""
import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path

UNITS_PER_SECOND = 1_000_000
# One microsecond is serialization precision, not permission to skip audio.
DURATION_TOLERANCE_US = 1


def to_us(seconds: float) -> int:
    if isinstance(seconds, bool) or not isinstance(seconds, (float, int)) or not math.isfinite(seconds):
        raise ValueError("Invalid finite audio offset")
    return round(seconds * UNITS_PER_SECOND)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SourceIdentity:
    path: str
    size_bytes: int
    mtime_ns: int
    fingerprint: str
    duration_us: int

    @classmethod
    def capture(cls, path: Path, duration: float) -> "SourceIdentity":
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("SOURCE_CHANGED_DURING_JOB")
        duration_us = to_us(duration)
        if duration_us <= 0:
            raise ValueError("Invalid source duration")
        return cls(str(path.resolve()), after.st_size, after.st_mtime_ns, digest, duration_us)

    def assert_unchanged(self) -> None:
        stat = Path(self.path).stat()
        if (stat.st_size, stat.st_mtime_ns) != (self.size_bytes, self.mtime_ns):
            raise ValueError("SOURCE_CHANGED_DURING_JOB")

    def to_dict(self) -> dict:
        return asdict(self)

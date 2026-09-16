"""Plausible tail coverage, independent of provider STOP and text fidelity.

Energy activity is not speech recognition. This gate never modifies timestamps
and cannot establish acoustic completeness within the transcribed interval.
"""
from dataclasses import asdict, dataclass
from enum import Enum
import math
import re
import subprocess

from src.block_builder import _find_binary


class CoverageDecision(str, Enum):
    PASS = "COVERAGE_PASS"
    WARNING = "COVERAGE_WARNING"
    RETRY = "COVERAGE_RETRY"
    FAIL = "COVERAGE_FAIL"


@dataclass(frozen=True)
class CoverageResult:
    decision: CoverageDecision
    last_transcript_timestamp: str | None
    tail_gap_seconds: float
    tail_activity: str
    active_tail_seconds: float | None = None
    reason: str = ""
    tail_gap_ratio: float | None = None
    physical_duration_seconds: float | None = None

    @property
    def allows_confirmation(self):
        return self.decision in (CoverageDecision.PASS, CoverageDecision.WARNING)

    def to_dict(self):
        return asdict(self)


class TailActivityAnalyzer:
    def analyze(self, path, relative_start, duration, threshold_db):
        """Return non-silent seconds; unreadable/incomplete analysis is not silence."""
        result = subprocess.run([
            _find_binary("ffmpeg"), "-hide_banner", "-nostats", "-ss", str(relative_start),
            "-t", str(duration), "-i", str(path), "-vn",
            "-af", f"silencedetect=noise={threshold_db}dB:d=0.5", "-progress", "pipe:1",
            "-f", "null", "-"], capture_output=True, text=True, timeout=120)
        progress = re.findall(r"out_time_us=(\d+)", result.stdout)
        if result.returncode or not progress or int(progress[-1]) / 1e6 < duration - 1.0:
            raise ValueError("Audio tail analysis failed or decoded interval is incomplete")
        silent, pending = 0.0, None
        for kind, value in re.findall(r"silence_(start|end):\s*([\d.]+)", result.stderr):
            value = min(duration, max(0.0, float(value)))
            if kind == "start":
                pending = value
            elif pending is not None:
                silent += max(0, value - pending)
                pending = None
        if pending is not None:
            silent += duration - pending
        return max(0.0, duration - silent)


class CoverageValidator:
    def __init__(self, config, analyzer=None):
        self.config = config
        self.analyzer = analyzer or TailActivityAnalyzer()
        self.max_ratio = getattr(config, "coverage_small_gap_max_ratio", 0.1)
        if not math.isfinite(self.max_ratio) or not 0 < self.max_ratio < 1:
            raise ValueError("Coverage small-gap ratio must be between zero and one")
        values = (config.coverage_tail_gap_threshold_sec, config.coverage_active_tail_min_sec,
                  config.coverage_silence_threshold_db, config.coverage_shrink_factor)
        if not all(math.isfinite(v) for v in values) or values[0] <= 0 or values[1] <= 0 or not 0 < values[3] < 1:
            raise ValueError("Invalid coverage thresholds or shrink factor")

    def validate(self, block, transcript):
        timestamps = []
        for segment in transcript.segments:
            if segment.timestamp:
                parts = [int(p) for p in segment.timestamp.split(":")]
                seconds = sum(v * m for v, m in zip(reversed(parts), (1, 60, 3600)))
                timestamps.append((seconds, segment.timestamp))
        last, original = max(timestamps) if timestamps else (block.start_time_seconds, None)
        evidence=getattr(transcript,"_wordinfo_evidence",None)
        if evidence and evidence.matches(transcript,block.start_time_seconds,block.end_time_seconds) and evidence.endpoint is not None:
            last=evidence.endpoint
            original=f"{int(last)//3600:02d}:{int(last)%3600//60:02d}:{int(last)%60:02d}"
        gap = max(0.0, block.end_time_seconds - last)
        duration = block.end_time_seconds - block.start_time_seconds
        if not math.isfinite(duration) or duration <= 0:
            return CoverageResult(CoverageDecision.FAIL, original, gap, "UNKNOWN", reason="Invalid physical duration")
        ratio = gap / duration
        metrics = dict(tail_gap_ratio=ratio, physical_duration_seconds=duration)
        if original is not None and gap <= self.config.coverage_tail_gap_threshold_sec and ratio <= self.max_ratio:
            return CoverageResult(CoverageDecision.PASS, original, gap, "NOT_ANALYZED_SMALL_GAP", **metrics)
        try:
            active = self.analyzer.analyze(block.file_path, max(0, last - block.start_time_seconds),
                                           gap, self.config.coverage_silence_threshold_db)
            if not math.isfinite(active) or not 0 <= active <= gap:
                raise ValueError("Invalid activity measurement")
        except Exception as exc:
            return CoverageResult(CoverageDecision.FAIL, original, gap, "UNKNOWN", reason=str(exc), **metrics)
        if active >= self.config.coverage_active_tail_min_sec:
            return CoverageResult(CoverageDecision.RETRY, original, gap, "ACTIVE", active,
                                  "Substantial activity after last timestamp; not proof of speech", **metrics)
        return CoverageResult(CoverageDecision.WARNING, original, gap, "MOSTLY_SILENT_OR_SPARSE", active, **metrics)

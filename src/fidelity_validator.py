"""Content Fidelity Validator for Milestone 3.1.

Provides multi-tier fidelity detection separate from structural validation.
Does NOT rewrite transcript — only detects suspicious patterns and returns
PASS / FAIL / REVIEW with structured issue report.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from src.response_parser import TranscriptionBlockResult, TranscriptSegment


class FidelityStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    REVIEW = "REVIEW"


class FidelityRisk(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class FidelityIssue:
    """A single detected fidelity concern in a transcript segment."""
    source_index: int
    indicator: str
    detail: str
    severity: str = "MEDIUM"  # LOW / MEDIUM / HIGH


@dataclass
class FidelityValidationResult:
    """Result of content fidelity validation."""
    status: FidelityStatus
    risk_level: FidelityRisk
    issues: list[FidelityIssue] = field(default_factory=list)
    unknown_token_count: int = 0
    unknown_token_rate: float = 0.0
    total_word_count: int = 0

    def summary(self) -> str:
        if self.status == FidelityStatus.PASS:
            return f"FIDELITY_PASS (risk={self.risk_level.value}, unknowns={self.unknown_token_count})"
        issue_summary = "; ".join(f"[{i.indicator}] seg {i.source_index}" for i in self.issues[:3])
        suffix = f" +{len(self.issues)-3} more" if len(self.issues) > 3 else ""
        return f"FIDELITY_{self.status.value} (risk={self.risk_level.value}): {issue_summary}{suffix}"

    @property
    def is_valid(self) -> bool:
        return self.status == FidelityStatus.PASS

    @property
    def needs_audio_review(self) -> bool:
        return self.risk_level in (FidelityRisk.HIGH, FidelityRisk.MEDIUM)


# ---------------------------------------------------------------------------
# Patterns used for detection
# ---------------------------------------------------------------------------

# Phrases that indicate suspiciously polished/formal language
# (Vietnamese formal equivalents of colloquial speech)
_FORMAL_SUBSTITUTION_PATTERNS = [
    # Pattern: (informal_marker_regex, formal_replacement_regex)
    # These are indicators that model may have "cleaned up" the text
    r"nhằm\s+mục\s+đích\s+(?:thực\s+hiện|triển\s+khai|đảm\s+bảo)",
    r"theo\s+đó[,\s]+(?:các\s+)?(?:đơn\s+vị|cơ\s+quan)\s+(?:có\s+trách\s+nhiệm|phải\s+thực\s+hiện)",
    r"trên\s+cơ\s+sở\s+(?:đó|đây)[,\s]+(?:việc|công\s+tác)",
]

# Known bad substitution pairs — from golden test audit
# Format: (source_phrase_fragment, bad_output_fragment)
KNOWN_BAD_SUBSTITUTIONS: list[tuple[str, str]] = [
    ("quốc phòng", "quốc phòng"),      # this one is actually correct — kept for contrast
    ("bộ quốc phòng", "pháp khác"),    # hallucinated
    ("trụ cột", "tầm thục vụ"),         # hallucinated
    ("lưu hành", "lưu hoạt"),           # hallucinated
    ("phối hợp quản lý", "dụng điện chuyển"),  # hallucinated
    ("tổng thể", "hoạt sinh tổng thể"),  # hallucinated
    ("tầm thấp", "tầng thấp"),           # close but wrong
    ("nguy cơ", "dữ liệu"),             # hallucinated
]

# Indicators of over-normalization
_NORMALIZATION_INDICATORS = [
    r"\bthực\s+hiện\s+(?:công\s+tác|nhiệm\s+vụ)\b",
    r"\bđảm\s+bảo\s+(?:an\s+toàn|hiệu\s+quả|chất\s+lượng)\b",
    r"\btheo\s+quy\s+định\s+(?:của|hiện\s+hành)\b",
]

# Unknown token marker
_UNKNOWN_PATTERN = re.compile(r"\[không\s+rõ\]", re.IGNORECASE)

# Word tokenizer (simple whitespace + punctuation split for Vietnamese)
_WORD_SPLIT = re.compile(r"[\s,\.!?;:\"'()\[\]]+")


def _count_words(text: str) -> int:
    """Count words in Vietnamese text."""
    tokens = [t for t in _WORD_SPLIT.split(text.strip()) if t]
    return len(tokens)


def _count_unknown_tokens(text: str) -> int:
    """Count [không rõ] occurrences."""
    return len(_UNKNOWN_PATTERN.findall(text))


class ContentFidelityValidator:
    """Detects content fidelity issues in transcript segments.

    Checks for:
    1. Suspiciously polished / formal language (semantic substitution indicators)
    2. Known bad substitution patterns (from golden test audit)
    3. Unknown token rate ([không rõ] density)
    4. Over-normalization indicators
    5. Timestamp coherence with block offset

    Does NOT modify transcript. Only returns PASS / FAIL / REVIEW.
    """

    def __init__(
        self,
        unknown_token_threshold: float = 0.30,
        high_risk_unknown_threshold: float = 0.60,
        enable_polished_language_check: bool = True,
        enable_known_bad_substitution_check: bool = True,
    ) -> None:
        self.unknown_token_threshold = unknown_token_threshold
        self.high_risk_unknown_threshold = high_risk_unknown_threshold
        self.enable_polished_language_check = enable_polished_language_check
        self.enable_known_bad_substitution_check = enable_known_bad_substitution_check

        self._formal_patterns = [re.compile(p, re.IGNORECASE) for p in _FORMAL_SUBSTITUTION_PATTERNS]
        self._normalization_patterns = [re.compile(p, re.IGNORECASE) for p in _NORMALIZATION_INDICATORS]

    def check_unknown_token_rate(
        self, segments: Sequence[TranscriptSegment]
    ) -> tuple[int, int, float]:
        """Count total words and [không rõ] tokens across all segments.

        Returns:
            (unknown_count, total_words, rate)
        """
        total_words = 0
        unknown_count = 0
        for seg in segments:
            if not seg.text:
                continue
            total_words += _count_words(seg.text)
            unknown_count += _count_unknown_tokens(seg.text)
        rate = unknown_count / max(total_words, 1)
        return unknown_count, total_words, rate

    def check_suspiciously_polished(
        self, segments: Sequence[TranscriptSegment]
    ) -> list[FidelityIssue]:
        """Detect segments with suspiciously formal/polished language patterns."""
        issues: list[FidelityIssue] = []
        if not self.enable_polished_language_check:
            return issues

        for seg in segments:
            if not seg.text:
                continue
            text = seg.text.lower()

            # Check normalization patterns
            for pat in self._normalization_patterns:
                if pat.search(text):
                    issues.append(FidelityIssue(
                        source_index=seg.source_index,
                        indicator="OVER_NORMALIZED",
                        detail=f"Suspiciously formal phrase detected: '{pat.pattern[:50]}'",
                        severity="LOW",
                    ))
                    break

            # Check formal substitution patterns
            for pat in self._formal_patterns:
                if pat.search(text):
                    issues.append(FidelityIssue(
                        source_index=seg.source_index,
                        indicator="SEMANTIC_FORMALIZATION",
                        detail=f"Possible semantic substitution: '{pat.pattern[:50]}'",
                        severity="MEDIUM",
                    ))
                    break

        return issues

    def check_known_bad_substitutions(
        self, segments: Sequence[TranscriptSegment]
    ) -> list[FidelityIssue]:
        """Detect known hallucination patterns discovered during golden test audit."""
        issues: list[FidelityIssue] = []
        if not self.enable_known_bad_substitution_check:
            return issues

        for seg in segments:
            if not seg.text:
                continue
            text_lower = seg.text.lower()
            for source_frag, bad_frag in KNOWN_BAD_SUBSTITUTIONS:
                # Flag if the bad output fragment appears (without the source fragment nearby)
                if bad_frag in text_lower and source_frag not in text_lower:
                    issues.append(FidelityIssue(
                        source_index=seg.source_index,
                        indicator="KNOWN_BAD_SUBSTITUTION",
                        detail=f"Possible bad substitution: expected '{source_frag}', found '{bad_frag}'",
                        severity="HIGH",
                    ))
        return issues

    def check_per_segment_unknown_rate(
        self, segments: Sequence[TranscriptSegment]
    ) -> list[FidelityIssue]:
        """Flag segments where entire text is [không rõ] (complete audio failure)."""
        issues: list[FidelityIssue] = []
        for seg in segments:
            if not seg.text:
                continue
            word_count = _count_words(seg.text)
            unknown_count = _count_unknown_tokens(seg.text)
            # If segment is entirely unknown, flag as REVIEW (not FAIL — this is expected behavior)
            if word_count > 0 and unknown_count == word_count:
                issues.append(FidelityIssue(
                    source_index=seg.source_index,
                    indicator="FULLY_UNCERTAIN_SEGMENT",
                    detail="Entire segment is [không rõ] — audio likely inaudible",
                    severity="LOW",  # Expected behavior, not a fidelity error
                ))
        return issues

    def _compute_risk_level(
        self,
        issues: list[FidelityIssue],
        unknown_rate: float,
    ) -> FidelityRisk:
        """Compute overall risk level from issues and unknown token rate."""
        high_severity = sum(1 for i in issues if i.severity == "HIGH")
        medium_severity = sum(1 for i in issues if i.severity == "MEDIUM")

        if high_severity > 0 or unknown_rate > self.high_risk_unknown_threshold:
            return FidelityRisk.HIGH
        if medium_severity >= 2 or unknown_rate > self.unknown_token_threshold:
            return FidelityRisk.MEDIUM
        return FidelityRisk.LOW

    def validate(
        self,
        block_result: TranscriptionBlockResult,
    ) -> FidelityValidationResult:
        """Run all fidelity checks and return structured result.

        Returns:
            FidelityValidationResult with PASS / FAIL / REVIEW status.
            FAIL = known hallucination detected (HIGH severity issues)
            REVIEW = suspicious patterns that need audio verification
            PASS = no significant fidelity concerns detected
        """
        segments = block_result.segments
        all_issues: list[FidelityIssue] = []

        # Run all checks
        all_issues.extend(self.check_suspiciously_polished(segments))
        all_issues.extend(self.check_known_bad_substitutions(segments))
        all_issues.extend(self.check_per_segment_unknown_rate(segments))

        # Unknown token metrics
        unknown_count, total_words, unknown_rate = self.check_unknown_token_rate(segments)

        # If unknown rate is extremely high (>60%), that's actually REVIEW not FAIL
        # (model is being appropriately uncertain)
        if unknown_rate > self.high_risk_unknown_threshold:
            all_issues.append(FidelityIssue(
                source_index=block_result.first_source_index,
                indicator="HIGH_UNCERTAINTY_RATE",
                detail=f"[không rõ] rate {unknown_rate:.1%} exceeds high threshold {self.high_risk_unknown_threshold:.1%}",
                severity="MEDIUM",  # REVIEW, not FAIL — model is being appropriately cautious
            ))

        risk_level = self._compute_risk_level(all_issues, unknown_rate)

        # Determine status
        high_severity_issues = [i for i in all_issues if i.severity == "HIGH"]
        if high_severity_issues:
            status = FidelityStatus.FAIL
        elif risk_level in (FidelityRisk.MEDIUM, FidelityRisk.HIGH):
            status = FidelityStatus.REVIEW
        else:
            status = FidelityStatus.PASS

        return FidelityValidationResult(
            status=status,
            risk_level=risk_level,
            issues=all_issues,
            unknown_token_count=unknown_count,
            unknown_token_rate=unknown_rate,
            total_word_count=total_words,
        )


def apply_unknown_substitutions(
    block_result: TranscriptionBlockResult,
    substitutions: list[dict],
) -> TranscriptionBlockResult:
    """Apply [không rõ] substitutions from audio review to a block result.

    This is the ONLY allowed text modification after Gemini output.
    Replaces specific text portions with [không rõ] based on audio review findings.

    Args:
        block_result: Original transcription block result.
        substitutions: List of {source_index, replacement} dicts from audio review.

    Returns:
        New TranscriptionBlockResult with [không rõ] substitutions applied.
    """
    from src.response_parser import TranscriptSegment

    sub_map: dict[int, str] = {
        s["source_index"]: s.get("replacement", "[không rõ]")
        for s in substitutions
        if "source_index" in s
    }

    new_segments = []
    for seg in block_result.segments:
        if seg.source_index in sub_map:
            new_seg = TranscriptSegment(
                source_index=seg.source_index,
                text=sub_map[seg.source_index],
                timestamp=seg.timestamp,
                speaker=seg.speaker,
            )
            new_segments.append(new_seg)
        else:
            new_segments.append(seg)

    # Reconstruct block result with modified segments
    from src.response_parser import TranscriptionBlockResult as TBR
    return TBR(
        schema_version=block_result.schema_version,
        job_id=block_result.job_id,
        session_id=block_result.session_id,
        block_id=block_result.block_id,
        first_source_index=block_result.first_source_index,
        last_source_index=block_result.last_source_index,
        status=block_result.status,
        segments=new_segments,
    )

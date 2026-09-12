"""Content Fidelity Validator for Milestone 3.1.

Provides heuristic fidelity detection separate from structural validation.
PASS means no issue detected, never acoustic verification. Trusted reference
text may establish a deterministic substitution; lexical presence cannot.
Does NOT rewrite transcript — only detects suspicious patterns and returns
PASS / FAIL / REVIEW with structured issue report.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

from src.response_parser import TranscriptionBlockResult, TranscriptSegment


class FidelityStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    REVIEW = "REVIEW"


class FidelityRisk(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class FidelityDecision(str, Enum):
    """Acceptance policy, separate from heuristic quality status."""
    ACCEPT = "ACCEPT"
    ACCEPT_WITH_WARNING = "ACCEPT_WITH_WARNING"
    RETRY_REVIEW = "RETRY_REVIEW"
    FAIL = "FAIL"


UNCERTAINTY_REASONS = frozenset({
    "UNCERTAIN_SPEECH", "FULLY_UNCERTAIN_SEGMENT", "HIGH_UNCERTAINTY_RATE",
})
DETERMINISTIC_REASONS = frozenset({
    "KNOWN_BAD_SUBSTITUTION", "TRUSTED_REFERENCE_MISMATCH", "UNSUPPORTED_REPLACEMENT",
})


@dataclass
class FidelityIssue:
    """A single detected fidelity concern in a transcript segment."""
    source_index: int
    indicator: str
    detail: str
    severity: str = "MEDIUM"  # LOW / MEDIUM / HIGH
    evidence: str = "HEURISTIC"  # HEURISTIC | TRUSTED_REFERENCE | AUDIO_REVIEW


@dataclass
class FidelityValidationResult:
    """Heuristic result, not a claim of acoustic correctness."""
    status: FidelityStatus
    risk_level: FidelityRisk
    issues: list[FidelityIssue] = field(default_factory=list)
    unknown_token_count: int = 0
    unknown_token_rate: float = 0.0
    total_word_count: int = 0
    source_mode: str = "UNKNOWN"
    has_trusted_reference: bool = False
    has_acoustic_review: bool = False

    def summary(self) -> str:
        if self.status == FidelityStatus.PASS:
            return f"FIDELITY_PASS (risk={self.risk_level.value}, unknowns={self.unknown_token_count})"
        issue_summary = "; ".join(f"[{i.indicator}] seg {i.source_index}" for i in self.issues[:3])
        suffix = f" +{len(self.issues)-3} more" if len(self.issues) > 3 else ""
        return f"FIDELITY_{self.status.value} (risk={self.risk_level.value}): {issue_summary}{suffix}"

    @property
    def is_valid(self) -> bool:
        """Quality-only compatibility property; use allows_confirmation for progress."""
        return self.status == FidelityStatus.PASS

    @property
    def decision(self) -> FidelityDecision:
        reasons = {issue.indicator for issue in self.issues}
        if self.status == FidelityStatus.FAIL or reasons & DETERMINISTIC_REASONS:
            return FidelityDecision.FAIL
        # Unknown review reasons fail closed, including REVIEW without any reason.
        informational = UNCERTAINTY_REASONS
        if (self.source_mode == "AUDIO" and not self.has_trusted_reference
                and not self.has_acoustic_review
                and all(i.evidence == "HEURISTIC" for i in self.issues if i.indicator == "OVER_NORMALIZED")):
            informational = informational | {"OVER_NORMALIZED"}
        if reasons - informational:
            return FidelityDecision.RETRY_REVIEW
        if reasons or (self.status == FidelityStatus.PASS and self.unknown_token_count):
            return FidelityDecision.ACCEPT_WITH_WARNING
        if self.status == FidelityStatus.REVIEW:
            return FidelityDecision.RETRY_REVIEW
        return FidelityDecision.ACCEPT

    @property
    def allows_confirmation(self) -> bool:
        return self.decision in (FidelityDecision.ACCEPT, FidelityDecision.ACCEPT_WITH_WARNING)

    @property
    def needs_audio_review(self) -> bool:
        return self.status != FidelityStatus.PASS


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
    """Count comparison units: each uncertainty marker is one semantic unit."""
    tokens = [t for t in _WORD_SPLIT.split(_UNKNOWN_PATTERN.sub(" ", text)) if t]
    return len(tokens) + _count_unknown_tokens(text)


def _count_unknown_tokens(text: str) -> int:
    """Count [không rõ] occurrences."""
    return len(_UNKNOWN_PATTERN.findall(text))


class ContentFidelityValidator:
    """Detects content fidelity issues in transcript segments.

    Checks for:
    1. Suspiciously polished / formal language (semantic substitution indicators)
    2. Known substitutions confirmed against caller-supplied trusted text
    3. Unknown token rate ([không rõ] density)
    4. Over-normalization indicators
    5. Long repeated phrases / extreme single-word loops

    Does NOT modify transcript. Only returns PASS / FAIL / REVIEW.
    """

    def __init__(
        self,
        unknown_token_threshold: float = 0.30,
        high_risk_unknown_threshold: float = 0.60,
        enable_polished_language_check: bool = True,
        enable_known_bad_substitution_check: bool = True,
        source_mode: str = "UNKNOWN",
    ) -> None:
        self.unknown_token_threshold = unknown_token_threshold
        self.high_risk_unknown_threshold = high_risk_unknown_threshold
        self.enable_polished_language_check = enable_polished_language_check
        self.enable_known_bad_substitution_check = enable_known_bad_substitution_check
        self.source_mode = source_mode

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
        self, segments: Sequence[TranscriptSegment],
        trusted_source_texts: Mapping[int, str] | None = None,
    ) -> list[FidelityIssue]:
        """Require a full-segment comparison with independently trusted text.

        The caller must never supply a model guess as a trusted reference.
        This establishes a text mismatch, not acoustic verification.
        """
        issues: list[FidelityIssue] = []
        if not self.enable_known_bad_substitution_check or trusted_source_texts is None:
            return issues

        for seg in segments:
            if not seg.text:
                continue
            text_lower = seg.text.lower()
            reference = trusted_source_texts.get(seg.source_index)
            if reference is None:
                continue
            reference = " ".join(reference.lower().split())
            text_lower = " ".join(text_lower.split())
            for source_frag, bad_frag in KNOWN_BAD_SUBSTITUTIONS:
                transformed = re.sub(r"(?<!\w)" + re.escape(source_frag) + r"(?!\w)", bad_frag, reference)
                if transformed != reference and transformed == text_lower:
                    issues.append(FidelityIssue(
                        source_index=seg.source_index,
                        indicator="KNOWN_BAD_SUBSTITUTION",
                        detail=f"Trusted text comparison confirms '{source_frag}' replaced by '{bad_frag}'",
                        severity="HIGH",
                    ))
        return issues

    def check_per_segment_unknown_rate(
        self, segments: Sequence[TranscriptSegment]
    ) -> list[FidelityIssue]:
        """Detect marker-only segments without assuming why audio is uncertain."""
        issues: list[FidelityIssue] = []
        for seg in segments:
            if not seg.text:
                continue
            unknown_count = _count_unknown_tokens(seg.text)
            # If segment is entirely unknown, flag as REVIEW (not FAIL — this is expected behavior)
            if unknown_count and not re.search(r"\w", _UNKNOWN_PATTERN.sub("", seg.text)):
                issues.append(FidelityIssue(
                    source_index=seg.source_index,
                    indicator="FULLY_UNCERTAIN_SEGMENT",
                    detail="Entire segment is uncertainty markers; audio review needed",
                    severity="LOW",  # Expected behavior, not a fidelity error
                ))
        return issues

    def check_repetition(self, segments: Sequence[TranscriptSegment]) -> list[FidelityIssue]:
        """Five consecutive copies of >=4 words, or 12 identical words, warrant review.

        These conservative thresholds indicate possible generation loops, not proof.
        Punctuation/case normalization is comparison-only; short stutters are valid.
        """
        issues = []
        for seg in segments:
            words = re.findall(r"\w+", _UNKNOWN_PATTERN.sub(" ", seg.text).casefold())
            looping = False
            for width, repeats in [(1, 12)] + [(n, 5) for n in range(4, len(words) // 5 + 1)]:
                for start in range(len(words) - width * repeats + 1):
                    phrase = words[start:start + width]
                    if words[start:start + width * repeats] == phrase * repeats:
                        looping = True
                        break
                if looping:
                    break
            if looping:
                issues.append(FidelityIssue(seg.source_index, "POSSIBLE_GENERATION_LOOP",
                                            "Repeated sequence requires audio review", "MEDIUM"))
        return issues

    def _compute_risk_level(
        self,
        issues: list[FidelityIssue],
        unknown_rate: float,
    ) -> FidelityRisk:
        """Compute overall risk level from issues and unknown token rate."""
        high_severity = sum(1 for i in issues if i.severity == "HIGH")

        if high_severity > 0 or unknown_rate > self.high_risk_unknown_threshold:
            return FidelityRisk.HIGH
        if issues or unknown_rate > self.unknown_token_threshold:
            return FidelityRisk.MEDIUM
        return FidelityRisk.LOW

    def validate(
        self,
        block_result: TranscriptionBlockResult,
        trusted_source_texts: Mapping[int, str] | None = None,
    ) -> FidelityValidationResult:
        """Run all fidelity checks and return structured result.

        Returns:
            FidelityValidationResult with PASS / FAIL / REVIEW status.
            FAIL = substitution confirmed against independently trusted source text
            REVIEW = suspicious patterns that need audio verification
            PASS = no significant fidelity concerns detected
        """
        segments = block_result.segments
        all_issues: list[FidelityIssue] = []

        # Run all checks
        all_issues.extend(self.check_suspiciously_polished(segments))
        all_issues.extend(self.check_known_bad_substitutions(segments, trusted_source_texts))
        # A narrowly proven filler/repetition removal is stronger than phrase style.
        # Keep it blocking even when the same indicator is informational in AUDIO.
        if trusted_source_texts:
            for seg in segments:
                reference = trusted_source_texts.get(seg.source_index)
                if reference is None:
                    continue
                original_words = re.findall(r"\w+", reference.casefold())
                reduced = []
                for word in original_words:
                    if word not in {"ờ", "ừ", "à"} and (not reduced or word != reduced[-1]):
                        reduced.append(word)
                if reduced != original_words and reduced == re.findall(r"\w+", seg.text.casefold()):
                    all_issues.append(FidelityIssue(seg.source_index, "OVER_NORMALIZED",
                        "Trusted reference comparison confirms removed fillers/repetition",
                        "MEDIUM", "TRUSTED_REFERENCE"))
        all_issues.extend(self.check_per_segment_unknown_rate(segments))
        all_issues.extend(self.check_repetition(segments))
        # Retain even sparse uncertainty as a quality signal without forcing retry.
        for seg in segments:
            if _count_unknown_tokens(seg.text):
                all_issues.append(FidelityIssue(seg.source_index, "UNCERTAIN_SPEECH",
                                                "Approved uncertainty marker retained", "LOW"))

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
        if any(i.indicator == "KNOWN_BAD_SUBSTITUTION" for i in all_issues):
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
            source_mode=self.source_mode,
            has_trusted_reference=bool(trusted_source_texts),
        )


def apply_unknown_substitutions(
    block_result: TranscriptionBlockResult,
    substitutions: list[dict],
) -> TranscriptionBlockResult:
    """Apply [không rõ] substitutions from audio review to a block result.

    Stage 2 permits only whole-segment uncertainty annotation from review.
    Partial edits and generated replacement speech are rejected. The caller must
    revalidate the returned block; annotation does not authorize confirmation.

    Args:
        block_result: Original transcription block result.
        substitutions: List of {source_index, replacement} dicts from audio review.

    Returns:
        New TranscriptionBlockResult with [không rõ] substitutions applied.
    """
    from src.response_parser import TranscriptSegment

    # Validate the entire batch before constructing any replacement.
    sub_map: dict[int, str] = {}
    indices = {seg.source_index for seg in block_result.segments}
    for sub in substitutions:
        index = sub.get("source_index")
        if type(index) is not int or index not in indices or index in sub_map:
            raise ValueError("Substitution requires a unique existing source_index")
        if sub.get("replacement", "[không rõ]") != "[không rõ]":
            raise ValueError("Only whole-segment [không rõ] annotation is permitted")
        sub_map[index] = "[không rõ]"

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

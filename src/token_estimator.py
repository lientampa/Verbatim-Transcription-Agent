"""Token estimation and dynamic output-ratio tracking module for DABB."""

import re
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class TokenBudgetReport:
    """Detailed breakdown of token budgets for a prospective block."""
    system_prompt_tokens: int
    control_tokens: int
    source_input_tokens: int
    total_input_tokens: int
    estimated_output_tokens: int
    total_context_tokens: int

    # Safety bounds
    context_limit: int
    max_output_tokens: int
    context_safety_margin: int
    output_safety_margin: int

    # Margins and decisions
    is_context_safe: bool
    is_output_safe: bool

    @property
    def is_safe(self) -> bool:
        return self.is_context_safe and self.is_output_safe


class TokenEstimator:
    """Accurate token estimator for text, segments, prompts, and structured JSON."""

    # Vietnamese syllables typically tokenize to ~1.2 - 1.5 tokens in BPE/SentencePiece
    VIETNAMESE_ACCENT_PATTERN = re.compile(
        r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]",
        re.IGNORECASE,
    )

    def estimate_text(self, text: str | None) -> int:
        """Estimate token count for arbitrary text.

        Uses linguistically informed character, syllable, and punctuation estimation
        tailored for multilingual/Vietnamese tokenizers.
        """
        if not text or not text.strip():
            return 0

        cleaned = text.strip()
        words = cleaned.split()
        word_count = len(words)

        # Count accented characters which cause BPE splits
        accented_chars = len(self.VIETNAMESE_ACCENT_PATTERN.findall(cleaned))
        punctuations = len(re.findall(r"[^\w\s]", cleaned))

        # Base estimation: ~1.2 tokens per word + punctuation + accent splits
        estimated = int(word_count * 1.25 + punctuations * 0.5 + accented_chars * 0.15)
        return max(1, estimated)

    def estimate_segment(self, segment: Any) -> int:
        """Estimate tokens for a SourceSegment including metadata framing overhead."""
        text = getattr(segment, "source_text", "") or getattr(segment, "text", "")
        speaker = getattr(segment, "speaker", "") or ""
        timestamp = getattr(segment, "timestamp", "") or ""

        content_tokens = self.estimate_text(text)
        metadata_tokens = self.estimate_text(speaker) + self.estimate_text(timestamp)
        # JSON formatting structure overhead (~12 tokens for {"source_index": N, "timestamp": ...})
        structure_overhead = 12

        return content_tokens + metadata_tokens + structure_overhead

    def estimate_system_prompt(self, prompt: str | None) -> int:
        """Estimate tokens consumed by system instruction."""
        return self.estimate_text(prompt)

    def estimate_control_tokens(self, control_info: str | dict | None) -> int:
        """Estimate tokens consumed by control directives and user prompt framing."""
        if isinstance(control_info, dict):
            import json
            text = json.dumps(control_info, ensure_ascii=False)
        else:
            text = str(control_info or "")
        return self.estimate_text(text) + 20  # Base envelope overhead

    def estimate_output_tokens(self, source_input_tokens: int, output_ratio: float) -> int:
        """Estimate generated output tokens based on input text tokens and ratio.

        Accounts for JSON envelope overhead (keys, schema metadata).
        """
        base_output = int(source_input_tokens * max(0.5, output_ratio))
        json_envelope_overhead = 80  # JSON root, schema_version, job_id, etc.
        return base_output + json_envelope_overhead

    def evaluate_budget(
        self,
        system_prompt: str,
        control_info: str | dict,
        segments: Sequence[Any],
        output_ratio: float,
        model_context_limit: int,
        model_max_output_tokens: int,
        context_safety_margin: int,
        output_safety_margin: int,
    ) -> TokenBudgetReport:
        """Perform comprehensive budget evaluation for a candidate block."""
        sys_tokens = self.estimate_system_prompt(system_prompt)
        ctrl_tokens = self.estimate_control_tokens(control_info)

        seg_tokens = sum(self.estimate_segment(s) for s in segments)
        total_input = sys_tokens + ctrl_tokens + seg_tokens

        est_output = self.estimate_output_tokens(seg_tokens, output_ratio)
        total_context = total_input + est_output

        safe_context_ceiling = model_context_limit - context_safety_margin
        safe_output_ceiling = model_max_output_tokens - output_safety_margin

        is_context_safe = total_context <= safe_context_ceiling
        is_output_safe = est_output <= safe_output_ceiling

        return TokenBudgetReport(
            system_prompt_tokens=sys_tokens,
            control_tokens=ctrl_tokens,
            source_input_tokens=seg_tokens,
            total_input_tokens=total_input,
            estimated_output_tokens=est_output,
            total_context_tokens=total_context,
            context_limit=model_context_limit,
            max_output_tokens=model_max_output_tokens,
            context_safety_margin=context_safety_margin,
            output_safety_margin=output_safety_margin,
            is_context_safe=is_context_safe,
            is_output_safe=is_output_safe,
        )


class OutputRatioTracker:
    """Tracks historical output/input token ratio using Exponential Moving Average (EMA)."""

    def __init__(self, default_ratio: float = 1.2, alpha: float = 0.3) -> None:
        self.default_ratio = max(0.1, default_ratio)
        self.alpha = min(1.0, max(0.01, alpha))
        self.current_ratio = self.default_ratio
        self.history: list[tuple[int, int]] = []

    def record_block(self, input_tokens: int, output_tokens: int) -> float:
        """Record actual input and output tokens of a completed block and update ratio."""
        if input_tokens <= 0 or output_tokens <= 0:
            return self.current_ratio

        self.history.append((input_tokens, output_tokens))
        observed_ratio = output_tokens / input_tokens

        # Clamp observed ratio to prevent crazy outliers (e.g. 0.1x - 5.0x)
        clamped_ratio = max(0.2, min(4.0, observed_ratio))

        if len(self.history) == 1:
            self.current_ratio = (self.default_ratio + clamped_ratio) / 2.0
        else:
            self.current_ratio = (self.alpha * clamped_ratio) + ((1.0 - self.alpha) * self.current_ratio)

        return self.current_ratio

    def get_ratio(self) -> float:
        """Return the current adaptive output/input ratio."""
        return self.current_ratio

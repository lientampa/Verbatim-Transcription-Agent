"""Audio-Grounded Review module for Milestone 3.1.

Sends a transcribed block back to Gemini with the original audio
to verify candidate transcript accuracy. Only confirms words actually
heard in audio; replaces unverifiable content with [không rõ].

This is intentionally a separate API call from transcription.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from src.fidelity_validator import FidelityValidationResult, FidelityRisk
from src.response_parser import TranscriptionBlockResult, TranscriptSegment

logger = logging.getLogger(__name__)


class ReviewStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNCERTAIN = "UNCERTAIN"
    SKIPPED = "SKIPPED"


@dataclass
class ReviewIssue:
    """An issue found during audio-grounded review."""
    source_index: int
    candidate_text: str
    reason: str
    replacement: str = "[không rõ]"


@dataclass
class AudioReviewResult:
    """Result of audio-grounded review call."""
    status: ReviewStatus
    issues: list[ReviewIssue] = field(default_factory=list)
    raw_response: str | None = None
    error: str | None = None

    @property
    def has_corrections(self) -> bool:
        return len(self.issues) > 0

    def get_substitutions(self) -> list[dict]:
        """Return substitution dicts for apply_unknown_substitutions()."""
        return [
            {"source_index": i.source_index, "replacement": i.replacement}
            for i in self.issues
        ]

    def summary(self) -> str:
        if self.status == ReviewStatus.SKIPPED:
            return "REVIEW_SKIPPED"
        if self.status == ReviewStatus.PASS:
            return f"REVIEW_PASS (0 corrections)"
        return (
            f"REVIEW_{self.status.value} "
            f"({len(self.issues)} corrections)"
        )


_REVIEW_SYSTEM_PROMPT = """Bạn là AUDIO VERIFICATION ENGINE — không phải transcription engine.

Nhiệm vụ: Nghe audio được cung cấp và xác minh từng segment trong candidate transcript.
Chỉ xác nhận những từ THỰC SỰ nghe được trong audio.
Nếu candidate chứa từ KHÔNG THỂ xác minh từ audio → thay bằng [không rõ].
KHÔNG sửa candidate bằng suy luận ngữ nghĩa.
KHÔNG rewrite để làm câu hay hơn.
KHÔNG thêm từ mà audio không có.

Output phải là JSON thuần túy theo schema:
{
  "status": "PASS | FAIL | UNCERTAIN",
  "issues": [
    {
      "source_index": <integer>,
      "candidate_text": "<text gốc>",
      "reason": "<lý do không xác minh được>",
      "replacement": "<text đã sửa với [không rõ] hoặc toàn bộ [không rõ]>"
    }
  ]
}

Nếu mọi segment đều xác minh được: trả về status PASS và issues=[].
Nếu có segment không xác minh được: trả về status FAIL và liệt kê issues.
Nếu audio quá ồn/không nghe được đủ để xác minh: trả về status UNCERTAIN."""


def _build_review_user_prompt(
    block_result: TranscriptionBlockResult,
    start_offset_seconds: float,
) -> str:
    """Build the user prompt for audio-grounded review."""
    offset_str = ""
    if start_offset_seconds > 0:
        minutes = int(start_offset_seconds // 60)
        seconds = int(start_offset_seconds % 60)
        offset_str = f"Audio block bắt đầu từ [{minutes:02d}:{seconds:02d}] trong file gốc.\n"

    segments_text = "\n".join(
        f"  source_index={seg.source_index}: \"{seg.text}\""
        for seg in block_result.segments
    )

    return (
        f"Nghe audio được đính kèm và xác minh candidate transcript sau đây.\n"
        f"{offset_str}"
        f"\nCANDIDATE TRANSCRIPT (block_id={block_result.block_id}):\n"
        f"{segments_text}\n\n"
        f"Trả về JSON xác minh theo schema đã được định nghĩa trong System Instruction.\n"
        f"TUYỆT ĐỐI KHÔNG rewrite hay polish. Chỉ xác minh và đánh dấu [không rõ] nếu không xác minh được."
    )


def _parse_review_response(raw_response: str) -> AudioReviewResult:
    """Parse Gemini review response into AudioReviewResult."""
    if not raw_response or not raw_response.strip():
        return AudioReviewResult(
            status=ReviewStatus.UNCERTAIN,
            error="Empty response from review model",
        )

    # Strip markdown if present
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()
    if not cleaned.startswith("{") and "{" in cleaned:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start:end + 1]

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning(f"[AudioReview] Failed to parse review JSON: {exc}")
        return AudioReviewResult(
            status=ReviewStatus.UNCERTAIN,
            raw_response=raw_response[:500],
            error=f"JSON parse error: {exc}",
        )

    status_str = str(data.get("status", "UNCERTAIN")).upper()
    try:
        status = ReviewStatus(status_str)
    except ValueError:
        status = ReviewStatus.UNCERTAIN

    issues = []
    for item in data.get("issues", []):
        if not isinstance(item, dict):
            continue
        try:
            issue = ReviewIssue(
                source_index=int(item.get("source_index", 0)),
                candidate_text=str(item.get("candidate_text", "")),
                reason=str(item.get("reason", "")),
                replacement=str(item.get("replacement", "[không rõ]")),
            )
            issues.append(issue)
        except (ValueError, TypeError):
            continue

    return AudioReviewResult(
        status=status,
        issues=issues,
        raw_response=raw_response[:500],
    )


class AudioGroundedReviewer:
    """Sends transcribed blocks back to Gemini for audio-grounded verification.

    Uses RISK_BASED mode by default: only reviews blocks with MEDIUM or HIGH
    fidelity risk. Can be configured to ALWAYS review all blocks.

    This is separate from the main transcription call and uses a different
    system prompt focused on verification rather than transcription.
    """

    REVIEW_MODE_RISK_BASED = "RISK_BASED"
    REVIEW_MODE_ALWAYS = "ALWAYS"
    REVIEW_MODE_DISABLED = "DISABLED"

    def __init__(
        self,
        gemini_client: Any,
        review_mode: str = REVIEW_MODE_RISK_BASED,
        review_model: str | None = None,
    ) -> None:
        """
        Args:
            gemini_client: GeminiClient instance for API calls.
            review_mode: RISK_BASED | ALWAYS | DISABLED
            review_model: Optional model override for review calls.
                          Defaults to same model as transcription.
        """
        self.gemini_client = gemini_client
        self.review_mode = review_mode.upper()
        self.review_model = review_model  # None = use same model as gemini_client

    def should_review(self, fidelity_result: FidelityValidationResult) -> bool:
        """Determine if this block warrants an audio recheck.

        RISK_BASED: Review if risk is MEDIUM or HIGH.
        ALWAYS: Review every block.
        DISABLED: Never review.
        """
        if self.review_mode == self.REVIEW_MODE_DISABLED:
            return False
        if self.review_mode == self.REVIEW_MODE_ALWAYS:
            return True
        # RISK_BASED (default)
        return fidelity_result.risk_level in (FidelityRisk.MEDIUM, FidelityRisk.HIGH)

    def review_block(
        self,
        gemini_file: Any,
        block_result: TranscriptionBlockResult,
        start_offset_seconds: float = 0.0,
    ) -> AudioReviewResult:
        """Send block back to Gemini for audio-grounded verification.

        Args:
            gemini_file: Uploaded Gemini file object (same audio block).
            block_result: Candidate transcript to verify.
            start_offset_seconds: Block start time for context.

        Returns:
            AudioReviewResult with PASS/FAIL/UNCERTAIN and any corrections.
        """
        user_prompt = _build_review_user_prompt(block_result, start_offset_seconds)

        model = self.review_model or self.gemini_client.model_name

        logger.info(
            f"[AudioReview] Reviewing block {block_result.block_id} "
            f"({len(block_result.segments)} segments, model={model})"
        )

        try:
            from google.genai import types
            config = types.GenerateContentConfig(
                system_instruction=_REVIEW_SYSTEM_PROMPT,
                temperature=0.0,
                response_mime_type="application/json",
            )
            raw_response = self.gemini_client._call_model(
                model=model,
                gemini_file=gemini_file,
                user_prompt=user_prompt,
                config=config,
            )
        except Exception as exc:
            logger.warning(f"[AudioReview] Review API call failed: {exc}")
            return AudioReviewResult(
                status=ReviewStatus.UNCERTAIN,
                error=f"Review API call failed: {exc}",
            )

        if not raw_response:
            return AudioReviewResult(
                status=ReviewStatus.UNCERTAIN,
                error="Empty response from review model",
            )

        result = _parse_review_response(raw_response)

        if result.status == ReviewStatus.FAIL:
            logger.warning(
                f"[AudioReview] Block {block_result.block_id} FAIL — "
                f"{len(result.issues)} corrections: "
                + ", ".join(
                    f"seg {i.source_index}: '{i.candidate_text[:30]}' → '{i.replacement[:30]}'"
                    for i in result.issues[:3]
                )
            )
        elif result.status == ReviewStatus.PASS:
            logger.info(f"[AudioReview] Block {block_result.block_id} PASS — no corrections needed.")
        else:
            logger.info(
                f"[AudioReview] Block {block_result.block_id} UNCERTAIN — "
                f"audio quality insufficient for full verification."
            )

        return result

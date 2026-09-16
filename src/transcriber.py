"""Transcriber module coordinating audio prompt, Gemini generation, and validation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.gemini_client import GeminiClient, GeminiClientError
from src.response_parser import (
    ResponseParser,
    TranscriptionBlockResult,
    TranscriptSegment,
    ResponseParserError,
    InvalidJSONError,
    SchemaValidationError,
)
from src.transcript_validator import TranscriptValidator, ValidationResult, ExpectedBlockContext
from src.fidelity_validator import ContentFidelityValidator, FidelityValidationResult, FidelityStatus
from src.validator import BaseValidator


class TranscriptionError(Exception):
    """Raised when transcription pipeline fails."""
    pass


@dataclass(frozen=True)
class TranscriptionOutcome:
    """Canonical block result; confirmation requires both validations to pass."""
    block_result: TranscriptionBlockResult
    structural_validation: ValidationResult
    fidelity_validation: FidelityValidationResult


class GeminiTranscriber:
    """Coordinates prompt loading, Gemini interaction, schema validation, and quality validation.

    Implements the Transcriber -> Structured JSON -> JSON Schema -> StructuralValidator
    -> ContentFidelityValidator pipeline for Milestone 3.1.
    """

    def __init__(
        self,
        gemini_client: GeminiClient,
        system_prompt_path: Path,
        validator: TranscriptValidator | None = None,
        response_parser: ResponseParser | None = None,
        schema_path: Path | None = None,
        fidelity_validator: ContentFidelityValidator | None = None,
    ) -> None:
        self.gemini_client = gemini_client
        self.system_prompt_path = Path(system_prompt_path)
        self.validator = validator or TranscriptValidator()
        self.response_parser = response_parser or ResponseParser(schema_path=schema_path)
        self.fidelity_validator = fidelity_validator or ContentFidelityValidator(source_mode="AUDIO")
        self._cached_prompt: str | None = None
        self.speaker_context: list[str] = []

    def load_system_prompt(self) -> str:
        """Load and cache the verbatim system instruction from file."""
        if self._cached_prompt is not None:
            return self._cached_prompt

        if not self.system_prompt_path.exists():
            raise TranscriptionError(
                f"System prompt file not found at: {self.system_prompt_path}"
            )

        try:
            content = self.system_prompt_path.read_text(encoding="utf-8").strip()
            if not content:
                raise TranscriptionError(f"System prompt file is empty: {self.system_prompt_path}")
            self._cached_prompt = content
            return self._cached_prompt
        except Exception as exc:
            raise TranscriptionError(f"Failed to read system prompt: {exc}") from exc

    def build_user_prompt(
        self,
        job_id: str,
        session_id: str,
        block_id: str,
        first_source_index: int,
        last_source_index: int | None = None,
        start_offset_seconds: float = 0.0,
        end_offset_seconds: float | None = None,
    ) -> str:
        """Construct structured user prompt instructing Gemini to return strict JSON."""
        def hhmmss(seconds):
            seconds = int(seconds)
            return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
        timestamp_contract = ""
        if self.fidelity_validator.source_mode == "AUDIO":
            timestamp_contract = (
                "All transcript timestamps are absolute from the beginning of the source file. Format: HH:MM:SS.\n"
                "20 minutes 26 seconds = 00:20:26, NOT 20:00:26 (which means 20 hours).\n"
                f"BLOCK_ABSOLUTE_START_SECONDS={start_offset_seconds}\n"
                f"BLOCK_ABSOLUTE_END_SECONDS={end_offset_seconds}\n"
                f"BLOCK_ABSOLUTE_START_HHMMSS={hhmmss(start_offset_seconds)}\n"
                f"BLOCK_ABSOLUTE_END_HHMMSS={hhmmss(end_offset_seconds) if end_offset_seconds is not None else 'UNSPECIFIED'}\n"
            )
            if end_offset_seconds is not None:
                width = end_offset_seconds - start_offset_seconds
                examples = [hhmmss(start_offset_seconds + min(3,width/4)),
                            hhmmss(start_offset_seconds + width * .36), hhmmss(max(start_offset_seconds,end_offset_seconds-2))]
                timestamp_contract += f"Valid timestamp examples inside this block (no transcript content): {', '.join(examples)}.\n"
                if end_offset_seconds < 72003:
                    timestamp_contract += "Invalid for this block: 20:00:03. Do not prepend block minutes as hours.\n"
        offset_note = ""
        if start_offset_seconds > 0:
            offset_str = hhmmss(start_offset_seconds)
            offset_note = (
                f"- Đoạn âm thanh này bắt đầu từ mốc thời gian thực [{offset_str}]. "
                f"Hãy đánh dấu timestamp của các câu nói dựa theo mốc thực này thay vì bắt đầu lại từ 00:00.\n"
            )

        range_note = ""
        if self.fidelity_validator.source_mode == "AUDIO":
            range_note = ("- AUDIO source_index là số thứ tự trong response, không phải định danh nguồn audio.\n"
                          "- first_source_index được gợi ý ở trên không ràng buộc ordinal với block trước.\n"
                          "- Dùng số nguyên dương, duy nhất, tăng dần; có thể bắt đầu lại ở mỗi block.\n"
                          "- first_source_index và last_source_index phải khớp segment đầu và cuối.\n")
        elif last_source_index is not None:
            range_note = f"- Bắt buộc đánh số source_index liên tục từ {first_source_index} đến {last_source_index}.\n"
        else:
            range_note = (
                f"- Đánh số source_index bắt đầu từ {first_source_index}, tăng dần liên tục (+1) cho từng segment.\n"
                f"- last_source_index trong JSON phải bằng source_index của segment cuối cùng.\n"
            )

        return (
            f"Hãy thực hiện phiên âm nguyên văn toàn bộ nội dung của block này theo đúng System Instruction.\n"
            f"Thông tin block:\n"
            f"- job_id: \"{job_id}\"\n"
            f"- session_id: \"{session_id}\"\n"
            f"- block_id: \"{block_id}\"\n"
            f"- first_source_index: {first_source_index}\n"
            f"{range_note}"
            f"- Nhãn người nói đã dùng trong các block được xác nhận: {self.speaker_context!r}. Giữ nhãn nếu có căn cứ cùng người; không suy đoán danh tính từ danh sách này.\n"
            f"{offset_note}"
            f"{timestamp_contract}"
            f"- Absolute audio boundaries (seconds): start={start_offset_seconds}, end={end_offset_seconds}.\n"
            f"LƯU Ý QUAN TRỌNG VỀ last_source_index: Giá trị \"last_source_index\" trong JSON BẮT BUỘC phải bằng chính xác source_index của segment cuối cùng trong mảng segments.\n"
            f"BẮT BUỘC trả về duy nhất một chuỗi JSON hợp lệ tuân thủ JSON Schema 1.0. "
            f"Tuyệt đối không dùng Markdown, không có lời giải thích, không trộn timestamp/speaker vào trường text."
        )

    def transcribe_block(
        self,
        gemini_file: Any,
        job_id: str,
        session_id: str,
        block_id: str,
        first_source_index: int,
        last_source_index: int | None = None,
        start_offset_seconds: float = 0.0,
        end_offset_seconds: float | None = None,
    ) -> TranscriptionOutcome:
        """Execute transcription, JSON parsing, schema validation, structural validation,
        and content fidelity validation for a block.

        Returns:
            TranscriptionOutcome containing the block and both validation results.

        Raises:
            TranscriptionError: If Gemini API call fails, JSON is invalid, or schema fails.
        """
        system_instruction = self.load_system_prompt()
        user_prompt = self.build_user_prompt(
            job_id=job_id,
            session_id=session_id,
            block_id=block_id,
            first_source_index=first_source_index,
            last_source_index=last_source_index,
            start_offset_seconds=start_offset_seconds,
            end_offset_seconds=end_offset_seconds,
        )

        try:
            raw_response = self.gemini_client.generate_transcription(
                gemini_file=gemini_file,
                system_instruction=system_instruction,
                user_prompt=user_prompt,
                response_mime_type="application/json",
            )
        except GeminiClientError as exc:
            raise TranscriptionError(f"Gemini API call failed: {exc}") from exc

        if not raw_response or not raw_response.strip():
            raise TranscriptionError("Received empty response from Gemini API.")

        # JSON Extraction + Schema Validation
        try:
            block_result = self.response_parser.parse(raw_response)
        except (InvalidJSONError, SchemaValidationError, ResponseParserError) as exc:
            raise TranscriptionError(f"Response parsing / schema validation failed: {exc}") from exc

        evidence=getattr(self.gemini_client,"last_wordinfo_evidence",None)
        native_order=bool(evidence and evidence.matches(block_result,start_offset_seconds,end_offset_seconds))
        if native_order:
            block_result._wordinfo_evidence=evidence
        # Structural Validation (Tier 1)
        val_result = self.validator.validate_block_result(
            block_result=block_result,
            expected_first_index=first_source_index,
            expected_last_index=last_source_index,
            expected_context=ExpectedBlockContext(job_id, session_id, block_id,
                None if self.fidelity_validator.source_mode == "AUDIO" else first_source_index,
                None if self.fidelity_validator.source_mode == "AUDIO" else last_source_index,
                start_offset_seconds, end_offset_seconds,
                source_mode=self.fidelity_validator.source_mode, native_wordinfo_order=native_order),
        )

        # Content Fidelity Validation (Tier 2) — always run, even if structural fails
        fidelity_result = self.fidelity_validator.validate(block_result)

        return TranscriptionOutcome(block_result, val_result, fidelity_result)

    def transcribe(self, gemini_file: Any, start_offset_seconds: float = 0.0) -> str:
        """Backward-compatible method returning raw string or formatted transcript."""
        outcome = self.transcribe_block(
            gemini_file=gemini_file,
            job_id="DEFAULT_JOB",
            session_id="DEFAULT_SESSION",
            block_id="BLOCK_001",
            first_source_index=1,
            start_offset_seconds=start_offset_seconds,
        )
        return "\n".join(s.text for s in outcome.block_result.segments)

    def transcribe_and_validate(
        self,
        gemini_file: Any,
        start_offset_seconds: float = 0.0,
    ) -> tuple[str, ValidationResult]:
        """Backward-compatible method returning (raw transcript string, ValidationResult)."""
        outcome = self.transcribe_block(
            gemini_file=gemini_file,
            job_id="DEFAULT_JOB",
            session_id="DEFAULT_SESSION",
            block_id="BLOCK_001",
            first_source_index=1,
            start_offset_seconds=start_offset_seconds,
        )
        text_lines = [
            f"[{s.timestamp} - {s.speaker}]: {s.text}" if s.timestamp and s.speaker else s.text
            for s in outcome.block_result.segments
        ]
        return "\n\n".join(text_lines), outcome.structural_validation

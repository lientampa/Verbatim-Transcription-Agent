"""Response parser and JSON schema validator for Gemini transcription output."""

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import jsonschema


class ResponseParserError(Exception):
    """Base error for response parsing failures."""
    pass


class InvalidJSONError(ResponseParserError):
    """Raised when the LLM response is not valid JSON."""
    pass


class SchemaValidationError(ResponseParserError):
    """Raised when JSON does not conform to the required JSON schema."""
    pass


@dataclass
class TranscriptSegment:
    """Individual transcript segment."""
    source_index: int
    text: str
    timestamp: str | None = None
    speaker: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptSegment":
        return cls(
            source_index=int(data["source_index"]),
            text=str(data["text"]).strip(),
            timestamp=data.get("timestamp"),
            speaker=data.get("speaker"),
        )


@dataclass
class TranscriptionBlockResult:
    """Structured transcription block result validated against schema."""
    schema_version: str
    job_id: str
    session_id: str
    block_id: str
    first_source_index: int
    last_source_index: int
    status: str
    segments: list[TranscriptSegment]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "session_id": self.session_id,
            "block_id": self.block_id,
            "first_source_index": self.first_source_index,
            "last_source_index": self.last_source_index,
            "status": self.status,
            "segments": [s.to_dict() for s in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptionBlockResult":
        segments = [TranscriptSegment.from_dict(s) for s in data.get("segments", [])]
        return cls(
            schema_version=data["schema_version"],
            job_id=data["job_id"],
            session_id=data["session_id"],
            block_id=data["block_id"],
            first_source_index=int(data["first_source_index"]),
            last_source_index=int(data["last_source_index"]),
            status=data["status"],
            segments=segments,
        )


def extract_json_string(raw_text: str) -> str:
    """Strip markdown formatting or surrounding noise to extract clean JSON text."""
    if not raw_text or not raw_text.strip():
        raise InvalidJSONError("Received empty response string.")

    cleaned = raw_text.strip()

    # If wrapped in markdown code fence ```json ... ``` or ``` ... ```
    if cleaned.startswith("```"):
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()
        else:
            # Strip outer backticks if present
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

    # If there are preamble or postamble texts around the JSON object
    if not cleaned.startswith("{") and "{" in cleaned:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]

    return cleaned


class ResponseParser:
    """Parses and validates Gemini structured JSON responses."""

    def __init__(self, schema_path: Path | None = None) -> None:
        if schema_path is None:
            # Default schema path relative to project root
            schema_path = Path(__file__).resolve().parent.parent / "schemas" / "transcription_result.schema.json"
        self.schema_path = Path(schema_path)
        self._schema_cache: dict[str, Any] | None = None

    def _load_schema(self) -> dict[str, Any]:
        if self._schema_cache is not None:
            return self._schema_cache

        if not self.schema_path.exists():
            raise ResponseParserError(f"JSON schema file not found at: {self.schema_path}")

        try:
            with open(self.schema_path, "r", encoding="utf-8") as f:
                self._schema_cache = json.load(f)
            return self._schema_cache
        except Exception as exc:
            raise ResponseParserError(f"Failed to read JSON schema: {exc}") from exc

    def parse(self, raw_text: str) -> TranscriptionBlockResult:
        """Parse raw LLM output into a validated TranscriptionBlockResult.

        Raises:
            InvalidJSONError: If raw_text cannot be parsed as JSON.
            SchemaValidationError: If parsed JSON violates schema.
        """
        json_str = extract_json_string(raw_text)

        try:
            parsed_data = json.loads(json_str)
        except Exception as exc:
            raise InvalidJSONError(f"Failed to parse JSON: {exc}. Raw snippet: {raw_text[:200]}") from exc

        if not isinstance(parsed_data, dict):
            raise InvalidJSONError(f"Expected JSON object, got {type(parsed_data).__name__}")

        schema = self._load_schema()
        try:
            jsonschema.validate(instance=parsed_data, schema=schema)
        except jsonschema.ValidationError as exc:
            raise SchemaValidationError(f"JSON schema validation failed: {exc.message} at path {list(exc.path)}") from exc

        # Synchronize last_source_index with actual last segment if present
        if parsed_data.get("segments") and isinstance(parsed_data["segments"], list) and len(parsed_data["segments"]) > 0:
            last_seg = parsed_data["segments"][-1]
            if isinstance(last_seg, dict) and "source_index" in last_seg:
                parsed_data["last_source_index"] = int(last_seg["source_index"])

        return TranscriptionBlockResult.from_dict(parsed_data)

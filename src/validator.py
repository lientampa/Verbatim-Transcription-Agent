"""Transcript Validator module for Milestone 3 quality assurance.

Re-exports TranscriptValidator and ValidationResult from transcript_validator.py,
with backward compatibility helper methods.
"""

from typing import Any
from src.transcript_validator import TranscriptValidator, ValidationResult
from src.response_parser import TranscriptionBlockResult, TranscriptSegment

__all__ = ["TranscriptValidator", "ValidationResult", "BaseValidator"]


class BaseValidator:
    """Abstract interface."""
    def validate_block_result(self, block_result: TranscriptionBlockResult) -> ValidationResult:
        raise NotImplementedError

    def validate(self, data: Any, *args: Any, **kwargs: Any) -> ValidationResult:
        raise NotImplementedError


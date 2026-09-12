"""Gemini API client wrapper for file upload and content generation."""

import time
import logging
import re
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from google.genai.errors import APIError

logger = logging.getLogger(__name__)


def _extract_retry_delay(exc: Exception, fallback_delay: float) -> float:
    """Extract server-recommended retry delay from 429 quota errors."""
    text = str(exc)
    # Check for 'Please retry in X.XXs'
    match_sec = re.search(r"retry in ([\d\.]+)s", text, re.IGNORECASE)
    if match_sec:
        try:
            return float(match_sec.group(1)) + 1.5
        except ValueError:
            pass

    # Check for 'retryDelay': 'Xs'
    match_delay = re.search(r"retryDelay':\s*'(\d+)s'", text, re.IGNORECASE)
    if match_delay:
        try:
            return float(match_delay.group(1)) + 1.5
        except ValueError:
            pass

    if "429" in text or "RESOURCE_EXHAUSTED" in text:
        return max(fallback_delay, 10.0)

    return fallback_delay


class GeminiClientError(Exception):
    """Base exception for Gemini client errors."""
    pass


class GeminiUploadError(GeminiClientError):
    """Raised when file upload to Gemini Files API fails."""
    pass


class GeminiTranscribeError(GeminiClientError):
    """Raised when Gemini content generation fails."""
    pass


class GeminiClient:
    """Wrapper around official Google GenAI SDK for audio transcription."""

    FALLBACK_MODEL = "gemini-3.5-flash"

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3.5-flash",
        max_retries: int = 3,
        initial_delay_seconds: float = 2.0,
        timeout_seconds: int = 300,
    ) -> None:
        if not api_key:
            raise GeminiClientError("API key must be provided to initialize GeminiClient.")
        self.model_name = model_name
        self.max_retries = max(1, max_retries)
        self.initial_delay_seconds = max(0.1, initial_delay_seconds)
        self.timeout_seconds = timeout_seconds
        self.client = genai.Client(api_key=api_key)

    def upload_audio(self, audio_path: Path) -> Any:
        """Upload an audio file to the Gemini Files API.

        Args:
            audio_path: Path to the local audio file.

        Returns:
            The uploaded Gemini File object.

        Raises:
            GeminiUploadError: If the upload fails or the file does not become ACTIVE.
        """
        if not audio_path.exists():
            raise GeminiUploadError(f"Audio file does not exist at: {audio_path}")

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                uploaded_file = self.client.files.upload(file=str(audio_path))

                # Poll until file is in ACTIVE state
                max_wait_seconds = 60
                start_time = time.time()
                while hasattr(uploaded_file, "state") and uploaded_file.state and uploaded_file.state.name == "PROCESSING":
                    if time.time() - start_time > max_wait_seconds:
                        raise GeminiUploadError("Timed out waiting for uploaded audio file to become ACTIVE.")
                    time.sleep(2)
                    uploaded_file = self.client.files.get(name=uploaded_file.name)

                if hasattr(uploaded_file, "state") and uploaded_file.state and uploaded_file.state.name == "FAILED":
                    raise GeminiUploadError(f"Uploaded audio file processing failed: {uploaded_file.error}")

                return uploaded_file
            except (APIError, ConnectionError, TimeoutError, OSError) as exc:
                last_err = exc
                if attempt < self.max_retries:
                    base_delay = self.initial_delay_seconds * (2 ** (attempt - 1))
                    delay = _extract_retry_delay(exc, base_delay)
                    logger.warning(
                        f"[GeminiClient] Upload failed (attempt {attempt}/{self.max_retries}). "
                        f"Retrying in {delay:.1f}s... Error: {exc}"
                    )
                    time.sleep(delay)
                else:
                    break

        raise GeminiUploadError(f"Failed to upload audio after {self.max_retries} attempts: {last_err}") from last_err

    def _call_model(self, model: str, gemini_file: Any, user_prompt: str, config: Any) -> str | None:
        """Helper to invoke models.generate_content and extract transcript text."""
        response = self.client.models.generate_content(
            model=model,
            contents=[gemini_file, user_prompt],
            config=config,
        )

        # 1. Try extracting text from candidates parts
        if response and hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and candidate.content and candidate.content.parts:
                extracted = "".join(
                    p.text for p in candidate.content.parts
                    if hasattr(p, "text") and p.text and not getattr(p, "thought", False)
                ).strip()
                if extracted:
                    return extracted

        # 2. Try standard response.text
        if response and response.text:
            return response.text.strip()

        return None

    def generate_transcription(
        self,
        gemini_file: Any,
        system_instruction: str,
        user_prompt: str = "Hãy thực hiện phiên âm nguyên văn toàn bộ nội dung âm thanh này theo đúng System Instruction.",
        response_mime_type: str = "application/json",
    ) -> str:
        """Call Gemini to transcribe audio using the provided system instruction.

        Includes automatic fallback to gemini-3.5-flash if the primary model is
        unavailable (503) or blocked (BlockedReason.OTHER).

        Args:
            gemini_file: The uploaded Gemini File object.
            system_instruction: Verbatim transcription instructions.
            user_prompt: User instruction prompt for Gemini.
            response_mime_type: Output MIME type ('application/json' by default).

        Returns:
            Verbatim transcript text.

        Raises:
            GeminiTranscribeError: If generation fails or response contains no text.
        """
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0,
            response_mime_type=response_mime_type,
        )

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                transcript = self._call_model(self.model_name, gemini_file, user_prompt, config)
                if transcript:
                    return transcript

                # If primary model returned empty response and is not the fallback model, try fallback
                if self.model_name != self.FALLBACK_MODEL:
                    logger.warning(
                        f"[GeminiClient] Model '{self.model_name}' returned empty response. "
                        f"Attempting fallback to '{self.FALLBACK_MODEL}'..."
                    )
                    fallback_transcript = self._call_model(self.FALLBACK_MODEL, gemini_file, user_prompt, config)
                    if fallback_transcript:
                        return fallback_transcript

                raise GeminiTranscribeError("Gemini returned an empty response or no text was generated.")

            except (APIError, ConnectionError, TimeoutError) as exc:
                last_err = exc
                err_str = str(exc)

                # If quota exhausted (429) or unavailable (503), try alternative models from pool
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str:
                    alternative_models = [m for m in ["gemini-3.6-flash", "gemini-3.8-flash", "gemini-3.5-flash-lite"] if m != self.model_name]
                    for alt_m in alternative_models:
                        try:
                            print(f"\n  [Quota Fallback] Đang chuyển sang model dự phòng '{alt_m}'...", end="", flush=True)
                            alt_transcript = self._call_model(alt_m, gemini_file, user_prompt, config)
                            if alt_transcript:
                                self.model_name = alt_m  # Persist the working model for remaining blocks
                                return alt_transcript
                        except Exception:
                            continue

                if attempt < self.max_retries:
                    base_delay = self.initial_delay_seconds * (2 ** (attempt - 1))
                    delay = _extract_retry_delay(exc, base_delay)
                    print(f" [Rate Limit] Đang đợi {delay:.1f}s trước khi thử lại ({attempt}/{self.max_retries})...", end="", flush=True)
                    time.sleep(delay)
                else:
                    break
            except Exception as exc:
                # If error occurred with primary model, try fallback before failing
                if self.model_name != self.FALLBACK_MODEL:
                    try:
                        fallback_transcript = self._call_model(self.FALLBACK_MODEL, gemini_file, user_prompt, config)
                        if fallback_transcript:
                            return fallback_transcript
                    except Exception:
                        pass
                raise GeminiTranscribeError(f"Transcription failed: {exc}") from exc

        raise GeminiTranscribeError(
            f"Failed to obtain transcript from Gemini after {self.max_retries} attempts: {last_err}"
        ) from last_err

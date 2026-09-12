"""Gemini API client wrapper for file upload and content generation."""

import time
import logging
import re
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from google.genai.errors import APIError
from src.failure_classifier import FailureType, classify_failure

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


def structured_retry_after(exc: Exception) -> float | None:
    """Read google.rpc.RetryInfo from the installed SDK's structured details."""
    import math
    def walk(value):
        if isinstance(value, dict):
            if str(value.get("@type", "")).endswith("google.rpc.RetryInfo"):
                delay = value.get("retryDelay", value.get("retry_delay"))
                try:
                    seconds = (float(delay.get("seconds", 0)) + float(delay.get("nanos", 0)) / 1e9
                               if isinstance(delay, dict) else float(str(delay).removesuffix("s")))
                    if math.isfinite(seconds) and seconds >= 0:
                        return seconds
                except (ValueError, TypeError):
                    pass
            for child in value.values():
                result = walk(child)
                if result is not None:
                    return result
        elif isinstance(value, list):
            for child in value:
                result = walk(child)
                if result is not None:
                    return result
        return None
    return walk(getattr(exc, "details", None))


class ModelReplanRequired(Exception):
    """Selected model needs a smaller physical asset before any provider call."""
    def __init__(self, target):
        self.target = target
        super().__init__(f"MODEL_REPLAN target={target}")


class GeminiClientError(Exception):
    """Base exception for Gemini client errors."""
    pass


class GeminiUploadError(GeminiClientError):
    """Raised when file upload to Gemini Files API fails."""
    pass


class GeminiTranscribeError(GeminiClientError):
    """Raised when Gemini content generation fails."""
    pass


class GeminiSizeError(GeminiClientError):
    """Provider-confirmed size failure; bypass same-asset client retries."""
    failure_type = FailureType.SIZE_FAILURE

    def __init__(self, message: str, metadata: dict | None = None):
        super().__init__(message)
        self.metadata = metadata or {}


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
        fallback_enabled: bool = True,
        fallback_policy: str = "PREFER_WAIT",
        max_backoff_seconds: float = 30.0,
        fallback_models: tuple[str, ...] = ("gemini-3.6-flash", "gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.5-flash"),
    ) -> None:
        if not api_key:
            raise GeminiClientError("API key must be provided to initialize GeminiClient.")
        self.model_name = model_name
        self.requested_model = model_name
        self.fallback_enabled = fallback_enabled
        if fallback_policy not in ("PREFER_WAIT", "PREFER_FALLBACK", "LOCK_MODEL") or not 0 <= max_backoff_seconds <= 60:
            raise GeminiClientError("Invalid fallback policy or maximum backoff (0..60 seconds)")
        self.fallback_policy = fallback_policy
        self.max_backoff_seconds = max_backoff_seconds
        self.fallback_models = fallback_models
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
                if classify_failure(exc) == FailureType.SIZE_FAILURE:
                    raise GeminiSizeError(str(exc), {"provider_error_code": getattr(exc, "code", None),
                                                    "provider_error_message": str(exc)}) from exc
                last_err = exc
                if attempt < self.max_retries:
                    base_delay = self.initial_delay_seconds * (2 ** (attempt - 1))
                    retry_after = structured_retry_after(exc)
                    delay = retry_after if retry_after is not None else _extract_retry_delay(exc, base_delay)
                    if delay > getattr(self, "max_backoff_seconds", 30.0):
                        break
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
        requested = getattr(self, "requested_model", getattr(self, "model_name", model))
        attempt = getattr(self, "_provider_attempt", 0) + 1
        self._provider_attempt = attempt
        block_match = re.search(r'- block_id: "([^"]+)"', user_prompt)
        self.last_response_metadata = {
            "requested_model": requested, "actual_model": model,
            "fallback_used": model != requested,
            "backoff_before_call_seconds": getattr(self, "_backoff_before_call", 0.0),
            "fallback_reason": (getattr(self, "_fallback_reason", None) or
                                getattr(self, "_sticky_fallback_reason", None)) if model != requested else None,
            "attempt": attempt, "provider_attempt": attempt, "retry_after_seconds": None, "block_id": block_match.group(1) if block_match else None,
        }
        self._backoff_before_call = 0.0
        if not hasattr(self, "call_history"):
            self.call_history = []
        self.call_history.append(self.last_response_metadata)
        print(f" [MODEL_CALL {self.last_response_metadata}]", end="", flush=True)
        try:
            response = self.client.models.generate_content(
                model=model, contents=[gemini_file, user_prompt], config=config)
        except Exception as exc:
            self.last_response_metadata.update(provider_error_code=getattr(exc, "code", None),
                                               provider_error_message=str(exc), retry_after_seconds=structured_retry_after(exc))
            if classify_failure(exc) == FailureType.SIZE_FAILURE:
                raise GeminiSizeError(str(exc), self.last_response_metadata) from exc
            raise
        usage = getattr(response, "usage_metadata", None)
        self.last_response_metadata.update({
            "input_tokens": getattr(usage, "prompt_token_count", None),
            "output_tokens": getattr(usage, "candidates_token_count", None),
            "total_tokens": getattr(usage, "total_token_count", None),
        })

        # 1. Try extracting text from candidates parts
        if response and hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            reason = getattr(candidate, "finish_reason", None)
            reason = getattr(reason, "value", reason)
            self.last_response_metadata["finish_reason"] = reason
            if reason == "MAX_TOKENS":
                raise GeminiSizeError("OUTPUT_TRUNCATED: provider finish_reason=MAX_TOKENS", self.last_response_metadata)
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

        Uses bounded same-model retries followed by a deterministic fallback plan.

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

        # One ordered plan per client/job. Once advanced, a candidate is never
        # reinserted; explicit same-model retries happen only before advancing.
        requested = getattr(self, "requested_model", self.model_name)
        policy = getattr(self, "fallback_policy", "PREFER_WAIT")
        locked = not getattr(self, "fallback_enabled", True) or policy == "LOCK_MODEL"
        fallbacks = getattr(self, "fallback_models", ("gemini-3.6-flash", "gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.5-flash"))
        plan = list(dict.fromkeys([requested] + ([] if locked else list(fallbacks))))
        current = self.model_name if self.model_name in plan and not locked else requested
        start = plan.index(current)
        last_err = None
        block_match = re.search(r'- block_id: "([^"]+)"', user_prompt)
        block_key = block_match.group(1) if block_match else None
        if block_key is None or block_key != getattr(self, "_attempt_block", None):
            self._provider_attempt = 0
        self._attempt_block = block_key
        self._fallback_reason = getattr(self, "_sticky_fallback_reason", None)
        if not hasattr(self, "_selection_reasons"):
            self._selection_reasons = {}
        for index, model in enumerate(plan[start:]):
            self._fallback_reason = str(last_err) if index else self._selection_reasons.get(model, self._fallback_reason)
            self._selection_reasons[model] = self._fallback_reason
            self._sticky_fallback_reason = self._fallback_reason
            self._backoff_before_call = 0.0
            self.model_name = model
            hook = getattr(self, "on_model_selected", None)
            if hook:
                hook(model, self._fallback_reason)
            for attempt in range(1, self.max_retries + 1):
                try:
                    transcript = self._call_model(model, gemini_file, user_prompt, config)
                    if transcript:
                        return transcript
                    raise GeminiTranscribeError("Gemini returned empty response")
                except GeminiSizeError:
                    raise
                except Exception as exc:
                    last_err = exc
                    retry_after = structured_retry_after(exc)
                    text = str(exc)
                    transient = isinstance(exc, (ConnectionError, TimeoutError)) or getattr(exc, "code", None) in (429, 503) or "429" in text or "503" in text
                    delay = retry_after if retry_after is not None else _extract_retry_delay(exc, self.initial_delay_seconds * 2 ** (attempt - 1))
                    maximum = getattr(self, "max_backoff_seconds", 30.0)
                    if transient and policy != "PREFER_FALLBACK" and attempt < self.max_retries and delay <= maximum:
                        print(f" [RETRY_SAME_MODEL model={model} provider_attempt={self._provider_attempt} retry_after_seconds={delay}]", flush=True)
                        time.sleep(delay)
                        self._backoff_before_call = delay
                        continue
                    break
        raise GeminiTranscribeError(f"Provider attempt plan exhausted: {last_err}") from last_err

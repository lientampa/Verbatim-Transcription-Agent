"""Gemini API client wrapper for file upload and content generation."""

import time
from src.model_policy import allowed_models, require_model_allowed, PREFERRED_MODEL_CHAIN, effective_model_chain
import logging
import re
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from google.genai.errors import APIError
from src.failure_classifier import FailureType, classify_failure

logger = logging.getLogger(__name__)


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


def retry_decision(client, exc, retry_index, fallback_available, retry_budget=None):
    """One bounded retry policy for uploads and generation; no sleeping here."""
    policy = getattr(client, "fallback_policy", "PREFER_WAIT")
    policy = {"PREFER_WAIT": "WAIT_FIRST", "LOCK_MODEL": "MODEL_LOCK"}.get(policy, policy)
    code = getattr(exc, "code", None)
    if code is None:
        match = re.search(r"\b(429|503)\b", str(exc))
        code = int(match.group()) if match else None
    provider_delay = structured_retry_after(exc)
    local = min(getattr(client, "retry_max_delay_seconds", 8.0),
                client.initial_delay_seconds * getattr(client, "backoff_multiplier", 2.0) ** retry_index)
    budget = getattr(client, "max_transient_retries", max(0, client.max_retries - 1))
    if retry_budget is not None:
        budget = min(budget, retry_budget)
    transient = code in (429, 503) or isinstance(exc, (ConnectionError, TimeoutError, OSError))
    delay = provider_delay if provider_delay is not None else local
    within_wait = provider_delay is None or provider_delay <= getattr(client, "max_backoff_seconds", 30.0)
    locked = policy == "MODEL_LOCK" or not getattr(client, "fallback_enabled", True)
    can_retry = transient and retry_index < budget and within_wait and policy != "PREFER_FALLBACK"
    if policy == "SPEED_FIRST" and code == 429 and provider_delay is None and fallback_available and not locked:
        can_retry = False
    action = "RETRY_SAME_MODEL" if can_retry else ("MODEL_FALLBACK" if fallback_available and not locked else "FAIL_PROVIDER")
    # A zero RetryInfo must not create a tight quota loop.
    if can_retry and delay <= 0:
        delay = max(0.1, local)
    return dict(action=action, policy=policy, error_code=code,
                provider_retry_after_seconds=provider_delay, local_backoff_seconds=local,
                actual_sleep_seconds=delay if can_retry else 0.0,
                reason="PROVIDER_DELAY_EXCEEDS_LIMIT" if not within_wait else
                       ("RETRY_BUDGET_EXHAUSTED" if retry_index >= budget else "RETRY_POLICY"))


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
        model_name: str = "gemini-3.8-flash",
        max_retries: int = 3,
        initial_delay_seconds: float = 1.0,
        timeout_seconds: int = 300,
        fallback_enabled: bool = True,
        allow_lite_models: bool = False,
        fallback_policy: str = "SPEED_FIRST",
        max_backoff_seconds: float = 5.0,
        retry_max_delay_seconds: float = 8.0,
        backoff_multiplier: float = 2.0,
        max_transient_retries: int = 2,
        fallback_models: tuple[str, ...] = PREFERRED_MODEL_CHAIN,
    ) -> None:
        if not api_key:
            raise GeminiClientError("API key must be provided to initialize GeminiClient.")
        require_model_allowed(model_name, allow_lite_models)
        self.allow_lite_models = False
        self.model_name = model_name
        self.requested_model = model_name
        self.fallback_enabled = fallback_enabled
        if fallback_policy not in ("PREFER_WAIT", "PREFER_FALLBACK", "LOCK_MODEL", "SPEED_FIRST", "WAIT_FIRST", "MODEL_LOCK") or not 0 <= max_backoff_seconds <= 60:
            raise GeminiClientError("Invalid fallback policy or maximum backoff (0..60 seconds)")
        if not 0 < retry_max_delay_seconds <= 60 or not 1 <= backoff_multiplier <= 4 or not 0 <= max_transient_retries <= 10:
            raise GeminiClientError("Invalid local backoff or retry budget")
        self.retry_max_delay_seconds = retry_max_delay_seconds
        self.backoff_multiplier = backoff_multiplier
        self.max_transient_retries = max_transient_retries
        self.fallback_policy = fallback_policy
        self.max_backoff_seconds = max_backoff_seconds
        self.fallback_models = tuple(allowed_models(fallback_models, allow_lite_models))
        self.max_retries = max(1, max_retries)
        self.initial_delay_seconds = max(0.1, initial_delay_seconds)
        self.timeout_seconds = timeout_seconds
        self.client = genai.Client(api_key=api_key)
        try:
            self.effective_model_chain = effective_model_chain(self.client.models.list())
        except Exception as exc:
            raise GeminiClientError("MODEL_DISCOVERY_FAILED: availability could not be verified") from exc
        locked = not fallback_enabled or fallback_policy in ("MODEL_LOCK", "LOCK_MODEL")
        if locked:
            self.effective_model_chain = tuple(m for m in self.effective_model_chain if m == model_name)
        print(f"[MODEL_POLICY] lite_allowed=false preferred_chain={list(PREFERRED_MODEL_CHAIN)} effective_chain={list(self.effective_model_chain)}", flush=True)
        if not self.effective_model_chain:
            print("[MODEL_FALLBACK_EXHAUSTED] reason=NO_ELIGIBLE_NON_LITE_MODEL", flush=True)
            raise GeminiClientError("NO_ELIGIBLE_MODEL: NO_ELIGIBLE_NON_LITE_MODEL")
        self.model_name = self.effective_model_chain[0]
        self.fallback_models = self.effective_model_chain
        if self.model_name != self.requested_model:
            self._sticky_fallback_reason = "DISCOVERED_PREFERRED_ORDER"


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
                decision = retry_decision(self, exc, attempt - 1, False, retry_budget=self.max_retries - 1)
                print(f" [PROVIDER_UPLOAD_RETRY {decision}]", flush=True)
                if decision["action"] != "RETRY_SAME_MODEL":
                    break
                time.sleep(decision["actual_sleep_seconds"])

        raise GeminiUploadError(f"Failed to upload audio after {self.max_retries} attempts: {last_err}") from last_err

    def refresh_audio_upload(self, gemini_file: Any, audio_path: Path) -> Any:
        """Validate an in-memory reference before reuse; never restore checkpoint file_id."""
        from datetime import datetime, timezone
        expiration = getattr(gemini_file, "expiration_time", None)
        if isinstance(expiration, datetime) and expiration.replace(tzinfo=expiration.tzinfo or timezone.utc) <= datetime.now(timezone.utc):
            return self.upload_audio(audio_path)
        try:
            refreshed = self.client.files.get(name=gemini_file.name)
        except Exception as exc:
            if getattr(exc, "code", None) == 404:
                return self.upload_audio(audio_path)
            raise GeminiUploadError("UPLOAD_REFERENCE_CHECK_FAILED") from exc
        state = getattr(getattr(refreshed, "state", None), "name", None)
        if state in ("FAILED", "EXPIRED"):
            return self.upload_audio(audio_path)
        return refreshed

    def _call_model(self, model: str, gemini_file: Any, user_prompt: str, config: Any) -> str | None:
        """Helper to invoke models.generate_content and extract transcript text."""
        require_model_allowed(model, getattr(self, "allow_lite_models", False))
        if hasattr(self, "effective_model_chain") and model not in self.effective_model_chain:
            raise GeminiTranscribeError("NO_ELIGIBLE_MODEL: model not discovered/supported")
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
            "attempt": attempt, "provider_attempt": attempt, "retry_after_seconds": None, "provider_retry_after_seconds": None, "block_id": block_match.group(1) if block_match else None,
        }
        self._backoff_before_call = 0.0
        if not hasattr(self, "call_history"):
            self.call_history = []
        self.call_history.append(self.last_response_metadata)
        console_metadata = dict(self.last_response_metadata)
        console_metadata["fallback_reason"] = str(console_metadata["fallback_reason"])[:180] if console_metadata["fallback_reason"] else None
        print(f" [MODEL_CALL {console_metadata}]", end="", flush=True)
        try:
            response = self.client.models.generate_content(
                model=model, contents=[gemini_file, user_prompt], config=config)
        except Exception as exc:
            self.last_response_metadata.update(provider_error_code=getattr(exc, "code", None),
                                               provider_error_message=str(exc), retry_after_seconds=structured_retry_after(exc), provider_retry_after_seconds=structured_retry_after(exc),
                                               provider_error_details=getattr(exc, "details", None))
            if classify_failure(exc) == FailureType.SIZE_FAILURE:
                raise GeminiSizeError(str(exc), self.last_response_metadata) from exc
            raise
        usage = getattr(response, "usage_metadata", None)
        self.last_response_metadata.update({
            "input_tokens": getattr(usage, "prompt_token_count", None),
            "output_tokens": getattr(usage, "candidates_token_count", None),
            "total_tokens": getattr(usage, "total_token_count", None),
            "thought_tokens": getattr(usage, "thoughts_token_count", None),
            "configured_max_output_tokens": getattr(config, "max_output_tokens", None),
        })

        # 1. Try extracting text from candidates parts
        if response and hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            reason = getattr(candidate, "finish_reason", None)
            reason = getattr(reason, "value", reason)
            self.last_response_metadata["finish_reason"] = reason
            if reason == "MAX_TOKENS":
                parts = getattr(getattr(candidate, "content", None), "parts", None) or []
                self.last_response_metadata["response_text_characters"] = sum(len(getattr(p, "text", None) or "") for p in parts if not getattr(p, "thought", False))
                diagnostic = {key: self.last_response_metadata.get(key) for key in (
                    "block_id", "actual_model", "provider_attempt", "finish_reason",
                    "input_tokens", "output_tokens", "thought_tokens", "total_tokens",
                    "configured_max_output_tokens", "response_text_characters")}
                print(f" [MAX_TOKENS_DIAGNOSTICS {diagnostic}]", flush=True)
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

    def advance_after_output_failure(self):
        """Advance once, forward only, to an eligible model; never change media."""
        health = self.model_health
        requested = getattr(self, "requested_model", self.model_name)
        if not getattr(self, "fallback_enabled", True) or getattr(self, "fallback_policy", "SPEED_FIRST") in ("LOCK_MODEL", "MODEL_LOCK"):
            raise GeminiTranscribeError("MODEL_OUTPUT_UNAVAILABLE: model locked and output circuit open")
        require_model_allowed(requested, getattr(self, "allow_lite_models", False))
        plan = allowed_models([requested] + list(getattr(self, "fallback_models", ())), getattr(self, "allow_lite_models", False))
        plan = list(getattr(self, "effective_model_chain", plan))
        current = self.model_name
        start = plan.index(current) + 1 if current in plan else len(plan)
        for model in plan[start:]:
            if health.eligible(model):
                reason = f"MODEL_HEALTH_CIRCUIT_OPEN:{current}"
                self.model_name = model
                self._sticky_fallback_reason = reason
                if not hasattr(self, "_selection_reasons"):
                    self._selection_reasons = {}
                self._selection_reasons[model] = reason
                print(f" [MODEL_HEALTH_SWITCH from={current} to={model}]", flush=True)
                return model
        raise GeminiTranscribeError(f"MODEL_OUTPUT_UNAVAILABLE NO_ELIGIBLE_FALLBACK_MODEL NO_ELIGIBLE_NON_LITE_MODEL: no eligible model after {current}; provider={health.provider_state}")

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
        locked = not getattr(self, "fallback_enabled", True) or policy in ("LOCK_MODEL", "MODEL_LOCK")
        fallbacks = getattr(self, "fallback_models", PREFERRED_MODEL_CHAIN)
        require_model_allowed(requested, getattr(self, "allow_lite_models", False))
        plan = allowed_models([requested] + ([] if locked else list(fallbacks)), getattr(self, "allow_lite_models", False))
        plan = list(getattr(self, "effective_model_chain", plan))
        current = self.model_name if self.model_name in plan and not locked else requested
        start = plan.index(current)
        last_err = None
        block_match = re.search(r'- block_id: "([^"]+)"', user_prompt)
        block_key = block_match.group(1) if block_match else None
        if block_key is None or block_key != getattr(self, "_attempt_block", None):
            self._provider_attempt = 0
        self._attempt_block = block_key
        health = getattr(self, "model_health", None)
        if health:
            health.begin_chain(block_key)
        self._fallback_reason = getattr(self, "_sticky_fallback_reason", None)
        if not hasattr(self, "_selection_reasons"):
            self._selection_reasons = {}
        for index, model in enumerate(plan[start:]):
            if health and not health.eligible(model, current=self.model_name):
                continue
            self._fallback_reason = str(last_err)[:180] if index else self._selection_reasons.get(model, self._fallback_reason)
            self._selection_reasons[model] = self._fallback_reason
            self._sticky_fallback_reason = self._fallback_reason
            self._backoff_before_call = 0.0
            self.model_name = model
            hook = getattr(self, "on_model_selected", None)
            if hook:
                hook(model, self._fallback_reason)
            if health:
                health.visited.add(model)
            for attempt in range(1, getattr(self, "max_transient_retries", self.max_retries - 1) + 2):
                try:
                    transcript = self._call_model(model, gemini_file, user_prompt, config)
                    if transcript:
                        if health:
                            health.provider_state.pop(model, None)
                        return transcript
                    raise GeminiTranscribeError("Gemini returned empty response")
                except GeminiSizeError:
                    raise
                except Exception as exc:
                    last_err = exc
                    next_model = next((m for m in plan[start + index + 1:] if not health or health.eligible(m)), None)
                    decision = retry_decision(self, exc, attempt - 1, next_model is not None)
                    if health:
                        health.provider_failure(model, decision)
                    self.last_response_metadata.update(decision)
                    print(f" [PROVIDER_RETRY model={model} provider_attempt={self._provider_attempt} {decision}]", flush=True)
                    if decision["action"] == "RETRY_SAME_MODEL":
                        time.sleep(decision["actual_sleep_seconds"])
                        self._backoff_before_call = decision["actual_sleep_seconds"]
                        continue
                    if decision["action"] == "MODEL_FALLBACK":
                        print(f" [MODEL_FALLBACK from={model} to={next_model} reason={decision['error_code']}_{decision['reason']} provider_retry_after_seconds={decision['provider_retry_after_seconds']} actual_sleep_seconds=0 policy={decision['policy']}]", flush=True)
                    if decision["action"] == "FAIL_PROVIDER":
                        print(" [MODEL_FALLBACK_EXHAUSTED reason=NO_ELIGIBLE_NON_LITE_MODEL]", flush=True)
                        raise GeminiTranscribeError(f"PROVIDER_UNAVAILABLE NO_ELIGIBLE_FALLBACK_MODEL NO_ELIGIBLE_NON_LITE_MODEL: {decision['reason']} code={decision['error_code']}") from exc
                    break
        raise GeminiTranscribeError("NO_ELIGIBLE_NON_LITE_MODEL: provider attempt plan exhausted") from last_err

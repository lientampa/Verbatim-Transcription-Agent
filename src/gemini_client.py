"""Gemini API client wrapper for file upload and content generation."""

import time
from src.model_policy import allowed_models, require_model_allowed, PREFERRED_MODEL_CHAIN, effective_model_chain, PRIMARY_MODEL
from src.provider_adapters import capabilities, GeminiGenerateContentAdapter, TranscriptionModelAdapter, ProviderAdapterError, AnnotationSemanticError
from src.runtime_models import RuntimeModels
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
    code = getattr(exc, "code", getattr(exc, "status_code", None))
    if code is None:
        match = re.search(r"\b(429|503)\b", str(exc))
        code = int(match.group()) if match else None
    provider_delay = structured_retry_after(exc)
    local = min(getattr(client, "retry_max_delay_seconds", 8.0),
                client.initial_delay_seconds * getattr(client, "backoff_multiplier", 2.0) ** retry_index)
    budget = getattr(client, "max_transient_retries", max(0, client.max_retries - 1))
    if retry_budget is not None:
        budget = min(budget, retry_budget)
    transient = code in (429, 503) if code is not None else isinstance(exc, (ConnectionError, TimeoutError, OSError))
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
        model_name: str = PRIMARY_MODEL,
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
        requested_max_output_tokens: int | None = None,
        transcription_thinking_level: str = "minimal",
        prompt_controlled_audio: bool = False,
        schema_path=None,
    ) -> None:
        if not api_key:
            raise GeminiClientError("API key must be provided to initialize GeminiClient.")
        require_model_allowed(model_name, allow_lite_models)
        if transcription_thinking_level not in ("minimal", "low", "medium", "high"):
            raise GeminiClientError("Invalid TRANSCRIPTION_THINKING_LEVEL")
        self.transcription_thinking_level = transcription_thinking_level
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
        self.capability_gate = None
        self.request_schema = None
        if prompt_controlled_audio:
            import json
            from pathlib import Path
            from src.provider_capability import CapabilityGate
            self.request_schema = json.loads(Path(schema_path or Path(__file__).resolve().parents[1] / 'schemas/transcription_result.schema.json').read_text(encoding='utf-8'))
            options = self.client._api_client._http_options
            endpoint = str(options.base_url) + '/' + str(options.api_version)
            self.capability_gate = CapabilityGate(self.client, self.request_schema, endpoint)
        try:
            discovered = list(self.client.models.list())
            self.adapters = {"generate_content": GeminiGenerateContentAdapter(), "transcribe": TranscriptionModelAdapter()}
            self.adapter_preflight = TranscriptionModelAdapter.preflight(self.client)
            self.effective_model_chain = effective_model_chain(discovered, self.adapter_preflight[0])
            self.model_output_limits = {m.name.removeprefix("models/"): m.output_token_limit for m in discovered if m.name}
            self.requested_max_output_tokens = requested_max_output_tokens
        except Exception as exc:
            raise GeminiClientError("MODEL_DISCOVERY_FAILED: availability could not be verified") from exc
        locked = not fallback_enabled or fallback_policy in ("MODEL_LOCK", "LOCK_MODEL")
        if not locked:
            self.requested_model = PRIMARY_MODEL
        if locked:
            self.effective_model_chain = tuple(m for m in self.effective_model_chain if m == model_name)
        self.preferred_chain = PREFERRED_MODEL_CHAIN
        self.discovered_chain = tuple(m for m in self.preferred_chain if any(d.name and d.name.removeprefix("models/") == m for d in discovered))
        self._runtime_models = RuntimeModels(self.preferred_chain, self.discovered_chain)
        self._preflight()
        eligible_chain = self.runtime_eligible_chain
        transient_paths = [p.model for p, e in self.capability_gate.results.items() if e.state == 'UNVERIFIED_TRANSIENT'] if self.capability_gate else []
        self.effective_primary_model = eligible_chain[0] if eligible_chain else None
        print(f"[MODEL_POLICY] lite_allowed=false configured_primary={self.requested_model} effective_primary={self.effective_primary_model} configured_chain={list(PREFERRED_MODEL_CHAIN)} eligible_chain={list(eligible_chain)} unverified_transient={transient_paths}", flush=True)
        if not self.runtime_eligible_chain:
            raise GeminiClientError("NO_ELIGIBLE_MODEL: " + self._exhaustion())
        self.model_name = self.runtime_eligible_chain[0]
        self.fallback_models = self.effective_model_chain
        if self.model_name != self.requested_model:
            self._sticky_fallback_reason = "DISCOVERED_PREFERRED_ORDER"


    def _registry(self):
        if not hasattr(self, "_runtime_models"):
            plan = getattr(self, "effective_model_chain", None) or tuple(allowed_models(
                [getattr(self, "requested_model", getattr(self, "model_name", PRIMARY_MODEL))] + list(getattr(self, "fallback_models", PREFERRED_MODEL_CHAIN))))
            self._runtime_models = RuntimeModels(plan, plan)
        health = getattr(self, "model_health", None)
        if health:
            for model, profile in health.profiles.items():
                state = self._runtime_models.states.get(model)
                if profile.degraded and not (state and (state.quarantined or state.reason in ("MODEL_STRUCTURAL_RETRY_EXHAUSTED", "MODEL_COVERAGE_RETRY_EXHAUSTED"))):
                    self._runtime_models.mark(model, "CIRCUIT_OPEN", "MODEL_HEALTH_CIRCUIT_OPEN", True)
        return self._runtime_models

    @property
    def runtime_eligible_chain(self):
        registry = self._registry()
        return tuple(m for m in registry.preferred if self._eligible(m, current=self.model_name if hasattr(self, "model_name") else None))

    def _eligible(self, model, current=None, *, reprobe=False):
        registry = self._registry()
        health = getattr(self, "model_health", None)
        eligible = registry.eligible(model) and (not health or health.eligible(model, current=current))
        gate = getattr(self, 'capability_gate', None)
        return eligible and (gate is None or gate.check(model, capabilities(model).provider_adapter, reprobe=reprobe).eligible)

    def _preflight(self):
        registry = self._runtime_models
        for model in self.discovered_chain:
            if getattr(self, 'capability_gate', None) is not None:
                print(f'[MODEL_DISCOVERED] model={model}', flush=True)
            if capabilities(model).provider_adapter == "transcribe":
                supported, reason = self.adapter_preflight
                if not supported:
                    registry.mark(model, "ADAPTER_UNAVAILABLE", "MODEL_ADAPTER_UNAVAILABLE", True, True, reason)
                    continue
            elif not callable(getattr(self.client.models, "generate_content", None)):
                registry.mark(model, "ADAPTER_UNAVAILABLE", "MODEL_ADAPTER_UNAVAILABLE", True, True, "GENERATE_CONTENT_ENDPOINT_UNAVAILABLE")
                continue
            if model not in self.effective_model_chain:
                registry.mark(model, "ADAPTER_UNAVAILABLE", "MODEL_ADAPTER_UNAVAILABLE", True,
                              detail="ENDPOINT_UNSUPPORTED_OR_MODEL_LOCK")
                continue
            if getattr(self, 'capability_gate', None) is not None:
                # Metadata access cannot establish generation endpoint access.
                continue  # runtime_eligible_chain below performs the audio contract probe.
            lookup = getattr(self.client.models, "get", None)
            if callable(lookup):
                try:
                    lookup(model=model)  # Cheap metadata only; no audio, generation or upload.
                    registry.mark(model, "DISCOVERED_UNVERIFIED", "MODEL_METADATA_ACCESS_CONFIRMED")
                except Exception as exc:
                    code = getattr(exc, "code", getattr(exc, "status_code", None))
                    if code in (403, 404):
                        registry.mark(model, "UNAVAILABLE", "MODEL_UNAVAILABLE", True, True, f"HTTP_{code}")
                    else:
                        registry.mark(model, "DISCOVERED_UNVERIFIED", "PREFLIGHT_ACCESS_UNVERIFIED", detail=type(exc).__name__)
        print(f"[MODEL_PREFLIGHT] states={registry.snapshot()}", flush=True)
        print(f"[RUNTIME_MODEL_CHAIN] discovered={list(self.discovered_chain)} eligible={list(self.runtime_eligible_chain)}", flush=True)

    def _record_failure(self, model, exc, decision):
        code = decision["error_code"]
        gate = getattr(self, 'capability_gate', None)
        if gate is not None:
            gate.invalidate(model, capabilities(model).provider_adapter, code)
            if code == 404:
                return 'ENDPOINT_MODEL_NOT_FOUND'  # Gate quarantine is path-scoped, not model-global.
        registry = self._registry()
        if code in (403, 404):
            reason = "MODEL_UNAVAILABLE"
            registry.mark(model, "UNAVAILABLE", reason, True, True, f"HTTP_{code}")
        elif isinstance(exc, ProviderAdapterError) and code is None:
            reason = "MODEL_ADAPTER_FAILURE"
            registry.mark(model, "ADAPTER_UNAVAILABLE", reason, True, True, getattr(exc, "reason", str(exc)).split(":")[0])
        else:
            blocked = decision["action"] != "RETRY_SAME_MODEL"
            reason = f"PROVIDER_{code}" if code else decision["failure_reason"]
            if blocked and code == 503:
                reason += "_RETRY_EXHAUSTED"
            registry.mark(model, "QUOTA_LIMITED" if code == 429 else "TRANSIENT_FAILURE", reason, blocked)
        if code == 429:
            delay = decision.get("provider_retry_after_seconds")
            if delay is None or delay <= 0:
                delay = getattr(self, "quota_cooldown_seconds", 30.0)
            registry.states[model].blocked_until = time.monotonic() + delay
        return reason

    def begin_fresh_block(self, block_key):
        if block_key is None or block_key == getattr(self, "_quota_block", None):
            return
        self._quota_block = block_key
        health = getattr(self, "model_health", None)
        if health:
            health.begin_chain(block_key)
        recovered = []
        for model, state in self._registry().states.items():
            if state.reason == "ANNOTATION_SEMANTIC_RETRY_EXHAUSTED" and not state.quarantined:
                self._runtime_models.mark(model, "ELIGIBLE" if state.proven_callable else "DISCOVERED_UNVERIFIED", "ANNOTATION_FRESH_BLOCK_PROBE")
                recovered.append(model)
            if state.state == "QUOTA_LIMITED" and not state.quarantined and state.blocked_until is not None and time.monotonic() >= state.blocked_until:
                self._runtime_models.mark(model, "ELIGIBLE" if state.proven_callable else "DISCOVERED_UNVERIFIED", "QUOTA_COOLDOWN_EXPIRED")
                if health:
                    health.provider_state.pop(model, None)
                recovered.append(model)
        if recovered and getattr(self, "fallback_enabled", True) and getattr(self, "fallback_policy", "SPEED_FIRST") not in ("MODEL_LOCK", "LOCK_MODEL"):
            plan = list(getattr(self, "effective_model_chain", self._registry().preferred))
            current = plan.index(self.model_name) if self.model_name in plan else len(plan)
            for model in plan[:current]:
                if model in recovered and self._eligible(model):
                    self.model_name = model
                    self._sticky_fallback_reason = "QUOTA_COOLDOWN_EXPIRED"
                    break

    def _exhaustion(self):
        snapshot = self._registry().snapshot()
        print(f"[MODEL_FALLBACK_EXHAUSTED] exclusions={snapshot}", flush=True)
        return "NO_ELIGIBLE_NON_LITE_MODEL: " + "; ".join(f"{model}={state['reason']}" for model, state in snapshot.items())

    def log_model_summary(self):
        states = self._registry().snapshot()
        used = list(dict.fromkeys(m.get("actual_model") for m in getattr(self, "call_history", []) if m.get("actual_model")))
        skipped = {m:s["reason"] for m,s in states.items() if s["blocked"] and m not in used}
        quarantined = {m:s["reason"] for m,s in states.items() if s["quarantined"]}
        print(f"[JOB_MODEL_SUMMARY] configured_primary={getattr(self, 'requested_model', PRIMARY_MODEL)} effective_primary={getattr(self, 'effective_primary_model', None)} actual_models_used={used} models_skipped={skipped} models_quarantined={quarantined} states={states}", flush=True)

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

    @staticmethod
    def minimum_thinking_level(model):
        return capabilities(model).preferred_transcription_thinking_level

    def generation_config(self, model, base):
        profile = capabilities(model)
        if profile.provider_adapter == "transcribe":
            return None
        minimum = self.minimum_thinking_level(model)
        level = getattr(self, "transcription_thinking_level", "minimal")
        if model in getattr(self, "_thinking_recovered_models", set()) or level == "minimal":
            level = minimum
        requested = getattr(self, "requested_max_output_tokens", None)
        discovered = getattr(self, "model_output_limits", {}).get(model)
        supported = discovered if type(discovered) is int and discovered > 0 else None
        effective = min(requested, supported) if requested is not None and supported else requested or supported
        return base.model_copy(update={
            "max_output_tokens": effective,
            "thinking_config": types.ThinkingConfig(thinking_level=level) if minimum else types.ThinkingConfig(thinking_budget=0) if model == "gemini-2.5-flash" else None,
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
            "tools": None,
        })

    def _call_with_config_recovery(self, model, asset, prompt, base):
        config = self.generation_config(model, base)
        try:
            return self._call_model(model, asset, prompt, config)
        except GeminiSizeError as exc:
            minimum = self.minimum_thinking_level(model)
            level = getattr(getattr(config, "thinking_config", None), "thinking_level", None)
            level = getattr(level, "value", level)
            if (exc.metadata.get("finish_reason") != "MAX_TOKENS" or not minimum or
                    str(level).lower() == minimum or getattr(self, "generation_config_retry_count", 0) >= 1):
                raise
            self.generation_config_retry_count = getattr(self, "generation_config_retry_count", 0) + 1
            self._thinking_recovered_models = getattr(self, "_thinking_recovered_models", set()) | {model}
            print(f" [GENERATION_CONFIG_RETRY] same_physical_block=true from_thinking_level={level} to_thinking_level={minimum} generation_config_retry_count={self.generation_config_retry_count}", flush=True)
            return self._call_model(model, asset, prompt, self.generation_config(model, base))

    def _call_model(self, model: str, gemini_file: Any, user_prompt: str, config: Any) -> str | None:
        """Helper to invoke models.generate_content and extract transcript text."""
        self.last_wordinfo_evidence=None
        require_model_allowed(model, getattr(self, "allow_lite_models", False))
        if hasattr(self, "effective_model_chain") and model not in self.effective_model_chain:
            raise GeminiTranscribeError("NO_ELIGIBLE_MODEL: model not discovered/supported")
        if hasattr(self, "_runtime_models") and not self._runtime_models.eligible(model):
            raise GeminiTranscribeError("MODEL_NOT_RUNTIME_ELIGIBLE: " + model)
        self.actual_capabilities = capabilities(model)
        gate = getattr(self, 'capability_gate', None)
        if gate is not None and not gate.check(model, self.actual_capabilities.provider_adapter).eligible:
            raise GeminiTranscribeError('REQUEST_PATH_UNSUPPORTED')
        requested = getattr(self, "requested_model", getattr(self, "model_name", model))
        attempt = getattr(self, "_provider_attempt", 0) + 1
        self._provider_attempt = attempt
        block_match = re.search(r'- block_id: "([^"]+)"', user_prompt)
        self.last_response_metadata = {
            "primary_model": PRIMARY_MODEL, "requested_model": requested, "actual_model": model,
            "fallback_index": PREFERRED_MODEL_CHAIN.index(model) if model in PREFERRED_MODEL_CHAIN else None,
            "runtime_model_state": self._registry().states.get(model).state if model in self._registry().states else "ELIGIBLE",
            "fallback_used": model != requested,
            "backoff_before_call_seconds": getattr(self, "_backoff_before_call", 0.0),
            "fallback_reason": (getattr(self, "_fallback_reason", None) or
                                getattr(self, "_sticky_fallback_reason", None)) if model != requested else None,
            "attempt": attempt, "provider_attempt": attempt, "retry_after_seconds": None, "provider_retry_after_seconds": None, "block_id": block_match.group(1) if block_match else None,
        }
        thinking = getattr(getattr(config, "thinking_config", None), "thinking_level", None)
        self.last_response_metadata.update(
            configured_max_output_tokens=getattr(self, "requested_max_output_tokens", None) or "UNSET",
            effective_max_output_tokens=(getattr(config, "max_output_tokens", None) or "PROVIDER_DEFAULT") if config is not None else "UNSUPPORTED_BY_ADAPTER",
            model_output_token_limit=getattr(self, "model_output_limits", {}).get(model) or "UNDISCOVERED",
            thinking_level=str(getattr(thinking, "value", thinking)).lower() if thinking else "NOT_APPLICABLE",
            generation_config_retry_count=getattr(self, "generation_config_retry_count", 0),
            provider_adapter=self.actual_capabilities.provider_adapter,
            thinking_budget=getattr(getattr(config, "thinking_config", None), "thinking_budget", None))
        print(" [GENERATION_CONFIG] " + " ".join(f"{k}={self.last_response_metadata[k]}" for k in
            ("block_id", "actual_model", "thinking_level", "effective_max_output_tokens", "model_output_token_limit")), flush=True)
        self._backoff_before_call = 0.0
        if not hasattr(self, "call_history"):
            self.call_history = []
        self.call_history.append(self.last_response_metadata)
        console_metadata = dict(self.last_response_metadata)
        # Console-only clarification: preserve persisted response metadata contract.
        console_metadata.pop('primary_model', None)
        console_metadata.update(configured_primary=requested,
                                effective_primary=getattr(self, 'effective_primary_model', None))
        if gate is not None:
            import hashlib
            instruction = getattr(config, 'system_instruction', None)
            attached = isinstance(instruction, str) and bool(instruction.strip())
            self.last_response_metadata.update(system_instruction_attached=attached,
                prompt_version=instruction.splitlines()[0] if attached else None,
                prompt_sha256=hashlib.sha256(instruction.encode('utf-8')).hexdigest() if attached else None,
                api_method=gate.path(model, self.actual_capabilities.provider_adapter).api_method)
            if not attached:
                raise GeminiTranscribeError('SYSTEM_INSTRUCTION_UNSUPPORTED')
            print('[SYSTEM_INSTRUCTION_DELIVERY] ' + str({k:self.last_response_metadata[k] for k in
                ('actual_model','provider_adapter','api_method','system_instruction_attached','prompt_version','prompt_sha256')}), flush=True)
        console_metadata["fallback_reason"] = str(console_metadata["fallback_reason"])[:180] if console_metadata["fallback_reason"] else None
        print(f" [MODEL_CALL {console_metadata}]", end="", flush=True)
        try:
            adapter = getattr(self, "adapters", {}).get(self.actual_capabilities.provider_adapter)
            if adapter is None:
                adapter = TranscriptionModelAdapter() if self.actual_capabilities.provider_adapter == "transcribe" else GeminiGenerateContentAdapter()
            response = adapter.generate(self.client, model, gemini_file, user_prompt, config)
        except Exception as exc:
            if getattr(exc,"provider_callable",False):
                state=self._registry().states.get(model)
                if state: state.provider_callable=state.proven_callable=True
            self.last_response_metadata.update(provider_error_code=getattr(exc, "code", None),
                                               provider_error_message=str(exc), retry_after_seconds=structured_retry_after(exc), provider_retry_after_seconds=structured_retry_after(exc),
                                               provider_error_details=getattr(exc, "details", None))
            if classify_failure(exc) == FailureType.SIZE_FAILURE:
                raise GeminiSizeError(str(exc), self.last_response_metadata) from exc
            raise
        state=self._registry().states.get(model)
        if state: state.provider_callable=state.proven_callable=True
        self.last_wordinfo_evidence=getattr(response,"wordinfo_evidence",None)
        if self.last_wordinfo_evidence is not None:
            evidence=self.last_wordinfo_evidence
            self.last_response_metadata.update(coverage_endpoint_seconds=evidence.endpoint,
                coverage_endpoint_basis="MAX_WORD_END_OFFSET", temporal_order_basis="PROVIDER_TEXT_ORDER",
                wordinfo_segment_hash=evidence.segment_hash)
        usage = getattr(response, "usage_metadata", None)
        self.last_response_metadata.update({
            "provider_usage_metadata": usage.model_dump(mode="json", exclude_none=True) if hasattr(usage, "model_dump") else (dict(vars(usage)) if usage is not None and hasattr(usage, "__dict__") else None),
            "physical_duration": getattr(self, "physical_duration", None),
            "dabb_target_input_tokens": getattr(self, "dabb_target_input_tokens", None),
            "provider_output_token_limit": getattr(self, "model_output_limits", {}).get(model),
            "input_tokens": getattr(usage, "prompt_token_count", None),
            "output_tokens": getattr(usage, "candidates_token_count", None),
            "total_tokens": getattr(usage, "total_token_count", None),
            "thought_tokens": getattr(usage, "thoughts_token_count", None),
            "visible_output_tokens": getattr(usage, "candidates_token_count", None),
        })

        # 1. Try extracting text from candidates parts
        if response and hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            reason = getattr(candidate, "finish_reason", None)
            reason = getattr(reason, "value", reason)
            self.last_response_metadata["finish_reason"] = reason
            if reason == "MAX_TOKENS":
                visible = self.last_response_metadata.get("output_tokens")
                thoughts = self.last_response_metadata.get("thought_tokens")
                ratio = thoughts / visible if isinstance(thoughts, (int, float)) and isinstance(visible, (int, float)) and visible > 0 else None
                pressure = isinstance(thoughts, (int, float)) and thoughts > 0 and isinstance(visible, (int, float)) and thoughts >= 4 * max(visible, 1)
                self.last_response_metadata.update(thought_to_visible_ratio=ratio,
                    max_tokens_classification="REASONING_OUTPUT_PRESSURE" if pressure else "OUTPUT_SIZE_PRESSURE")
                print(f" [MAX_TOKENS_CLASSIFICATION] type={self.last_response_metadata['max_tokens_classification']} thought_tokens={thoughts} output_tokens={visible} ratio={ratio if ratio is not None else 'UNAVAILABLE'}", flush=True)
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

    def advance_after_output_failure(self, failure_reason=None):
        """Advance once, forward only, to an eligible model; never change media."""
        health = self.model_health
        if not getattr(self, "model_name", None):
            raise GeminiTranscribeError("MODEL_OUTPUT_UNAVAILABLE NO_ELIGIBLE_FALLBACK_MODEL: actual model unavailable")
        if failure_reason:
            self._registry().mark(self.model_name, "DEGRADED" if failure_reason == "ANNOTATION_SEMANTIC_RETRY_EXHAUSTED" else "CIRCUIT_OPEN", failure_reason, True)
        requested = getattr(self, "requested_model", self.model_name)
        if not getattr(self, "fallback_enabled", True) or getattr(self, "fallback_policy", "SPEED_FIRST") in ("LOCK_MODEL", "MODEL_LOCK"):
            raise GeminiTranscribeError("MODEL_OUTPUT_UNAVAILABLE: model locked and output circuit open; " + self._exhaustion())
        require_model_allowed(requested, getattr(self, "allow_lite_models", False))
        plan = allowed_models([requested] + list(getattr(self, "fallback_models", ())), getattr(self, "allow_lite_models", False))
        plan = list(getattr(self, "effective_model_chain", plan))
        current = self.model_name
        start = plan.index(current) + 1 if current in plan else len(plan)
        for model in plan[start:]:
            if self._eligible(model, reprobe=True):
                reason = failure_reason or f"MODEL_HEALTH_CIRCUIT_OPEN:{current}"
                self.model_name = model
                self.actual_capabilities = capabilities(model)
                self._sticky_fallback_reason = reason
                if not hasattr(self, "_selection_reasons"):
                    self._selection_reasons = {}
                self._selection_reasons[model] = reason
                print(f" [MODEL_HEALTH_SWITCH from={current} to={model} reason={reason}]", flush=True)
                return model
        raise GeminiTranscribeError("MODEL_OUTPUT_UNAVAILABLE NO_ELIGIBLE_FALLBACK_MODEL: " + self._exhaustion())

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
            response_json_schema=getattr(self, 'request_schema', None),
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
        fresh_match = re.search(r'- block_id: "([^"]+)"', user_prompt)
        self.begin_fresh_block(fresh_match.group(1) if fresh_match else None)
        current = self.model_name if self.model_name in plan and not locked else requested
        start = plan.index(current)
        last_err = None
        block_match = re.search(r'- block_id: "([^"]+)"', user_prompt)
        block_key = block_match.group(1) if block_match else None
        if block_key is None or block_key != getattr(self, "_attempt_block", None):
            self._provider_attempt = 0
            self.generation_config_retry_count = 0
        self._attempt_block = block_key
        health = getattr(self, "model_health", None)
        if health:
            health.begin_chain(block_key)
        self._fallback_reason = getattr(self, "_sticky_fallback_reason", None)
        if not hasattr(self, "_selection_reasons"):
            self._selection_reasons = {}
        for index, model in enumerate(plan[start:]):
            if not self._eligible(model, current=self.model_name):
                continue
            self._fallback_reason = str(last_err)[:180] if index else self._selection_reasons.get(model, self._fallback_reason)
            self._selection_reasons[model] = self._fallback_reason
            self._sticky_fallback_reason = self._fallback_reason
            self._backoff_before_call = 0.0
            self.model_name = model
            self.actual_capabilities = capabilities(model)
            if getattr(self, 'capability_gate', None) is not None:
                print(f'[MODEL_SELECTED] model={model} adapter={self.actual_capabilities.provider_adapter} eligibility_reason=CAPABILITY_ELIGIBLE routing_reason={self._fallback_reason or "INITIAL_SELECTION"}', flush=True)
            hook = getattr(self, "on_model_selected", None)
            if hook:
                hook(model, self._fallback_reason)
            if health:
                health.visited.add(model)
            for attempt in range(1, getattr(self, "max_transient_retries", self.max_retries - 1) + 2):
                try:
                    transcript = self._call_with_config_recovery(model, gemini_file, user_prompt, config)
                    if transcript:
                        self._registry().mark(model, "ELIGIBLE", "LAST_CALL_SUCCEEDED")
                        if health:
                            health.provider_state.pop(model, None)
                        return transcript
                    raise GeminiTranscribeError("EMPTY_RESPONSE: Gemini returned empty response")
                except AnnotationSemanticError:
                    raise  # Model-output budget is owned by the orchestration layer.
                except GeminiSizeError:
                    raise
                except Exception as exc:
                    last_err = exc
                    next_model = next((m for m in plan[start + index + 1:] if self._eligible(m)), None)
                    # Re-probe only when existing policy would leave this model.
                    # Never feed full audio to a transient-unverified fallback.
                    intent = retry_decision(self, exc, attempt - 1, True)
                    if intent['action'] == 'MODEL_FALLBACK':
                        next_model = next((m for m in plan[start + index + 1:]
                                           if self._eligible(m, reprobe=True)), None)
                    decision = retry_decision(self, exc, attempt - 1, next_model is not None)
                    decision["failure_reason"] = ("EMPTY_RESPONSE" if "EMPTY_RESPONSE" in str(exc) else
                        f"PROVIDER_{decision['error_code']}" if decision["error_code"] is not None else type(exc).__name__)
                    decision["failure_reason"] = self._record_failure(model, exc, decision)
                    if health:
                        health.provider_failure(model, decision)
                    self.last_response_metadata.update(decision)
                    print(f" [PROVIDER_RETRY model={model} provider_attempt={self._provider_attempt} {decision}]", flush=True)
                    if decision["action"] == "RETRY_SAME_MODEL":
                        time.sleep(decision["actual_sleep_seconds"])
                        self._backoff_before_call = decision["actual_sleep_seconds"]
                        continue
                    if decision["action"] == "MODEL_FALLBACK":
                        print(f" [MODEL_FALLBACK from={model} to={next_model} reason={decision['failure_reason']} provider_retry_after_seconds={decision['provider_retry_after_seconds']} actual_sleep_seconds=0 policy={decision['policy']}]", flush=True)
                    if decision["action"] == "FAIL_PROVIDER":
                        raise GeminiTranscribeError("PROVIDER_UNAVAILABLE NO_ELIGIBLE_FALLBACK_MODEL: " + self._exhaustion()) from exc
                    break
        raise GeminiTranscribeError(self._exhaustion()) from last_err

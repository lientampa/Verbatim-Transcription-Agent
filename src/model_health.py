"""Job-scoped output health and separate provider eligibility; no automatic reprobes."""
from dataclasses import dataclass, asdict
import time
from src.model_policy import is_model_allowed, require_model_allowed


@dataclass
class ModelHealthProfile:
    severe_coverage_failures: int = 0
    timestamp_semantic_failures: int = 0
    structural_generation_failures: int = 0
    successful_blocks: int = 0
    successful_duration: float = 0
    consecutive_strikes: int = 0
    last_failure_reason: str | None = None
    degraded: bool = False


class ModelHealth:
    def __init__(self, config):
        if not 0 < config.severe_coverage_ratio <= 1 or config.model_health_strike_limit < 1:
            raise ValueError("Invalid model health thresholds")
        self.allow_lite_models = getattr(config, "allow_lite_models", False)
        self.ratio = config.severe_coverage_ratio
        self.limit = config.model_health_strike_limit
        self.profiles = {}
        self.provider_state = {}
        self.diagnostics = []
        self.chain = None
        self.visited = set()

    def profile(self, model):
        require_model_allowed(model, self.allow_lite_models)
        return self.profiles.setdefault(model, ModelHealthProfile())

    def begin_chain(self, block_id):
        if block_id != self.chain:
            self.chain = block_id
            self.visited = set()

    def eligible(self, model, current=None):
        if not is_model_allowed(model, self.allow_lite_models):
            return False
        state = self.provider_state.get(model, {})
        return (not self.profile(model).degraded
                and not state.get("provider_retry_exhausted", False)
                and time.monotonic() >= state.get("provider_cooldown_until", 0)
                and (model == current or model not in self.visited))

    def provider_failure(self, model, decision):
        state = self.provider_state.setdefault(model, {})
        delay = decision.get("provider_retry_after_seconds")
        if decision.get("error_code") == 429 and delay is not None:
            state["provider_cooldown_until"] = time.monotonic() + delay
        if decision["action"] != "RETRY_SAME_MODEL" and decision.get("error_code") != 429:
            state["provider_retry_exhausted"] = True
        state["last_provider_reason"] = decision["reason"]

    def _strike(self, model, reason):
        profile = self.profile(model)
        profile.consecutive_strikes += 1
        profile.last_failure_reason = reason
        profile.degraded = profile.consecutive_strikes >= self.limit
        print(f" [MODEL_HEALTH model={model} {asdict(profile)}]", flush=True)
        if profile.degraded:
            print(f" [MODEL_HEALTH_CIRCUIT_OPEN model={model} reason={reason}]", flush=True)
        return profile.degraded

    def coverage_failure(self, model, coverage):
        if (model and coverage.decision.value == "COVERAGE_RETRY" and coverage.tail_activity == "ACTIVE"
                and coverage.tail_gap_ratio is not None and coverage.tail_gap_ratio >= self.ratio):
            self.profile(model).severe_coverage_failures += 1
            return self._strike(model, "REPEATED_SEVERE_COVERAGE")
        return False

    def structural_failure(self, model, result, block, transcript):
        if not model:
            return False
        self.profile(model).structural_generation_failures += 1
        self.diagnostics.append(dict(model=model, block_id=transcript.block_id,
                                     errors=list(result.errors), response=transcript.to_dict()))
        # Count one strike per response, not per segment. Ordinary drift is not
        # enough: this catches hour/minute confusion at least an hour out of range.
        impossible = False
        if "TIMESTAMP_OUT_OF_RANGE" in result.reason_codes:
            for segment in transcript.segments:
                try:
                    seconds = sum(int(v)*m for v,m in zip(reversed(segment.timestamp.split(":")), (1,60,3600)))
                    impossible |= seconds > block.end_time_seconds + 3600 or seconds < block.start_time_seconds - 3600
                except (ValueError, AttributeError):
                    continue
        if set(result.reason_codes) & {"TIMESTAMP_OUT_OF_RANGE", "TIMESTAMP_ORDER_ERROR"}:
            self.profile(model).timestamp_semantic_failures += 1
        if impossible:
            return self._strike(model, "REPEATED_TIMESTAMP_SEMANTICS")
        return False

    def success(self, model, duration):
        if model:
            profile = self.profile(model)
            profile.successful_blocks += 1
            profile.successful_duration += duration
            profile.consecutive_strikes = 0

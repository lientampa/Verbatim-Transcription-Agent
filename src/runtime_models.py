"""Job-local observed eligibility; never edits configured preference."""
from dataclasses import dataclass, asdict
from enum import Enum


class RuntimeModelState(str, Enum):
    DISCOVERED_UNVERIFIED="DISCOVERED_UNVERIFIED"
    ELIGIBLE="ELIGIBLE"
    UNAVAILABLE="UNAVAILABLE"
    ADAPTER_UNAVAILABLE="ADAPTER_UNAVAILABLE"
    TRANSIENT_FAILURE="TRANSIENT_FAILURE"
    DEGRADED="DEGRADED"
    CIRCUIT_OPEN="CIRCUIT_OPEN"
    QUOTA_LIMITED="QUOTA_LIMITED"


@dataclass
class ModelState:
    state: str="DISCOVERED_UNVERIFIED"
    reason: str="DISCOVERED_ACCESS_UNVERIFIED"
    blocked: bool=False
    quarantined: bool=False
    detail: str | None=None


class RuntimeModels:
    def __init__(self, preferred, discovered):
        self.preferred=tuple(preferred)
        self.states={m:ModelState() if m in discovered else ModelState("UNAVAILABLE","NOT_DISCOVERED",True) for m in preferred}

    def mark(self,model,state,reason,blocked=False,quarantined=False,detail=None):
        previous=self.states.get(model)
        if previous and previous.quarantined and not quarantined:
            return  # Only a fresh job can clear confirmed quarantine.
        self.states[model]=ModelState(state,reason,blocked,quarantined,detail)
        if quarantined and (previous is None or not previous.quarantined):
            print(f"[MODEL_QUARANTINE] model={model} scope=JOB reason={reason}",flush=True)

    def eligible(self,model):
        return model in self.states and not self.states[model].blocked

    def snapshot(self):
        return {model:asdict(state) for model,state in self.states.items()}

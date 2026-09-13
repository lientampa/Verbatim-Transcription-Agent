"""Product eligibility, shared by configuration, selection, profiles and SDK calls."""


class ModelDisallowedError(ValueError):
    pass


def is_model_allowed(model_name: str, allow_lite_models: bool = False) -> bool:
    return "-lite" not in model_name.lower()


def require_model_allowed(model_name: str, allow_lite_models: bool = False) -> None:
    if not is_model_allowed(model_name, allow_lite_models):
        raise ModelDisallowedError(f"MODEL_DISALLOWED: LITE_MODEL_DISABLED: {model_name}")


def allowed_models(candidates, allow_lite_models: bool = False):
    return list(dict.fromkeys(m for m in candidates if is_model_allowed(m, allow_lite_models)))


PREFERRED_MODEL_CHAIN = ("gemini-2.5-flash", "gemini-3.5-transcribe", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.8-flash")
PRIMARY_MODEL = PREFERRED_MODEL_CHAIN[0]


def effective_model_chain(models, transcribe_available=False):
    """Fixed order intersected with account discovery and adapter endpoint support."""
    available = set()
    for model in models:
        if not model.name:
            continue
        name = model.name.removeprefix("models/")
        actions = set(model.supported_actions or [])
        if name == "gemini-3.5-transcribe":
            # Exact model has a documented Interactions contract. Models.list
            # action strings do not define its request schema; require local SDK support.
            if transcribe_available:
                available.add(name)
        elif "generateContent" in actions:
            available.add(name)
    return tuple(m for m in PREFERRED_MODEL_CHAIN if m in available and is_model_allowed(m))

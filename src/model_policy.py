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


PREFERRED_MODEL_CHAIN = ("gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash")


def effective_model_chain(models):
    """Intersect the fixed preference order with discovered generation support."""
    available = {m.name.removeprefix("models/") for m in models
                 if m.name and "generateContent" in (m.supported_actions or [])}
    return tuple(m for m in PREFERRED_MODEL_CHAIN if m in available and is_model_allowed(m))

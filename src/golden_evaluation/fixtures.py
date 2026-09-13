"""Read-only reference validation and private source binding."""
import json
import math
from pathlib import Path
from copy import deepcopy
import jsonschema
from src.source_identity import sha256_file

REVIEWED = {"HUMAN_REVIEWED", "DOUBLE_REVIEWED"}
SCHEMA = json.loads((Path(__file__).resolve().parents[2] / "schemas" / "golden_reference.schema.json").read_text(encoding="utf-8"))


class GoldenFixtureError(ValueError):
    pass


def validate_reference(reference, source_fingerprint=None):
    try:
        jsonschema.Draft202012Validator(SCHEMA).validate(reference)
        duration = reference["source"]["duration_seconds"]
        if not math.isfinite(duration):
            raise ValueError("duration must be finite")
        last = -1
        for segment in reference["segments"]:
            start = segment["start_seconds"]
            if not math.isfinite(start) or not last <= start <= duration:
                raise ValueError("invalid timestamp order/bounds")
            last = start
            if not segment["speaker"].strip() or not segment["text"].strip():
                raise ValueError("empty speaker/text")
            if segment["uncertain"] != ("[không rõ]" in segment["text"]):
                raise ValueError("uncertainty metadata contradicts marker")
            for event in segment.get("events", []):
                if event["text"] not in segment["text"]:
                    raise ValueError("event absent from reference text")
            if any(name not in segment["text"] for name in segment.get("proper_names", [])):
                raise ValueError("proper name absent from reference text")
        for region in reference.get("regions", []):
            start, end = region["start_seconds"], region["end_seconds"]
            if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= duration:
                raise ValueError("invalid region")
    except (jsonschema.ValidationError, ValueError, TypeError, KeyError) as exc:
        raise GoldenFixtureError(f"GOLDEN_FIXTURE_INVALID: {exc.message if isinstance(exc, jsonschema.ValidationError) else exc}") from exc
    if source_fingerprint is not None and source_fingerprint != reference["source"]["fingerprint_sha256"]:
        raise GoldenFixtureError("GOLDEN_SOURCE_MISMATCH")
    return deepcopy(reference)


def load_fixture(path):
    path = Path(path)
    try:
        reference = validate_reference(json.loads(path.read_text(encoding="utf-8")))
        source = reference["source"]
        if source.get("audio_path"):
            audio = Path(source["audio_path"])
            audio = audio if audio.is_absolute() else path.parent / audio
            validate_reference(reference, sha256_file(audio))
        elif reference["review"]["status"] in REVIEWED:
            raise GoldenFixtureError("GOLDEN_FIXTURE_INVALID: reviewed fixture requires locally verifiable audio_path")
        return reference
    except (OSError, json.JSONDecodeError) as exc:
        raise GoldenFixtureError(f"GOLDEN_FIXTURE_INVALID: {type(exc).__name__}") from exc


def canonical_segments(data, duration):
    """Accept stored canonical start timestamps; never read SRT estimated ends."""
    if not isinstance(data, (list, tuple)):
        raise GoldenFixtureError("MODEL_TRANSCRIPT_INVALID: segments must be an array")
    result, previous = [], -1
    for item in data:
        if hasattr(item, "to_dict"):
            item = item.to_dict()
        if not isinstance(item, dict):
            raise GoldenFixtureError("MODEL_TRANSCRIPT_INVALID: segment must be an object")
        entry = deepcopy(item)
        start = entry.get("start_seconds")
        if start is None:
            parts = str(entry.get("timestamp", "")).split(":")
            if len(parts) not in (2, 3):
                raise GoldenFixtureError("MODEL_TRANSCRIPT_INVALID: absolute start required")
            try:
                values = list(map(float, parts))
                if not 0 <= values[-1] < 60 or (len(values)==3 and not 0 <= values[-2] < 60):
                    raise ValueError()
                start = sum(value * (60 ** power) for power, value in enumerate(reversed(values)))
            except ValueError as exc:
                raise GoldenFixtureError("MODEL_TRANSCRIPT_INVALID: timestamp") from exc
        if isinstance(start, bool) or not isinstance(start, (float,int)) or not math.isfinite(start) or not 0 <= start <= duration or start < previous:
            raise GoldenFixtureError("MODEL_TRANSCRIPT_INVALID: timestamp bounds/order")
        if not isinstance(entry.get("text"), str) or not entry["text"].strip() or not isinstance(entry.get("speaker"), str) or not entry["speaker"].strip():
            raise GoldenFixtureError("MODEL_TRANSCRIPT_INVALID: text/speaker")
        entry["start_seconds"] = start
        result.append(entry)
        previous = start
    return result

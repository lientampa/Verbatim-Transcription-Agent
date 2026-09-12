"""Structural validation preserves model evidence, including invalid metadata."""
from copy import deepcopy
import json

import pytest

from src.response_parser import ResponseParser, SchemaValidationError
from src.transcript_validator import ExpectedBlockContext, TranscriptValidator


def raw():
    return dict(schema_version="1.0", job_id="JOB_001", session_id="SESSION_001",
                block_id="BLOCK_001", first_source_index=1, last_source_index=1,
                status="CONFIRMED", segments=[dict(source_index=1, text="  Tôi... tôi [không rõ].  ",
                                                    timestamp="10:02", speaker=" Người A ")])


def context(**kwargs):
    values = dict(job_id="JOB_001", session_id="SESSION_001", block_id="BLOCK_001",
                  first_source_index=1, audio_start=600, audio_end=900)
    values.update(kwargs)
    return ExpectedBlockContext(**values)


def validate_preserved(data, expected=None):
    before = deepcopy(data)
    parsed = ResponseParser().parse(json.dumps(data, ensure_ascii=False))
    assert parsed.to_dict() == before
    snapshot = deepcopy(parsed)
    result = TranscriptValidator().validate(parsed, expected_context=expected or context())
    assert parsed == snapshot
    assert data == before
    return result


def test_valid_matching_context_preserves_every_field():
    assert validate_preserved(raw()).is_valid


def test_parser_and_validator_preserve_wrong_end():
    data = raw()
    data["last_source_index"] = 99
    result = validate_preserved(data)
    assert not result.is_valid
    assert "LAST_SOURCE_INDEX_MISMATCH" in result.reason_codes


def test_identity_errors_aggregate_without_repair():
    data = raw()
    for field in ("job_id", "session_id", "block_id"):
        data[field] = "WRONG"
    result = validate_preserved(data)
    assert not result.is_valid
    assert set(result.reason_codes) == {"JOB_ID_MISMATCH", "SESSION_ID_MISMATCH", "BLOCK_ID_MISMATCH"}


@pytest.mark.parametrize("timestamp,reason", [
    ("99:99", "INVALID_TIMESTAMP"), ("00:99:01", "INVALID_TIMESTAMP"),
    ("12:70:00", "INVALID_TIMESTAMP"), ("abc", "INVALID_TIMESTAMP"),
    (" 10:02", "INVALID_TIMESTAMP"), ("20:00", "TIMESTAMP_OUT_OF_RANGE"),
    ("09:58", "TIMESTAMP_OUT_OF_RANGE"),
])
def test_timestamp_rejection_preserves_value(timestamp, reason):
    data = raw()
    data["segments"][0]["timestamp"] = timestamp
    assert reason in validate_preserved(data).reason_codes


@pytest.mark.parametrize("timestamp", [None, "10:00", "15:00", "00:10:02"])
def test_optional_and_boundary_timestamps_pass(timestamp):
    data = raw()
    data["segments"][0]["timestamp"] = timestamp
    assert validate_preserved(data).is_valid


def test_fractional_slice_boundary_uses_documented_one_second_tolerance():
    data = raw()
    data["segments"][0]["timestamp"] = "10:00"
    assert validate_preserved(data, context(audio_start=600.75)).is_valid
    assert not validate_preserved(data, context(audio_start=600.75, timestamp_tolerance=0)).is_valid


def test_timestamp_order_rejected_without_sorting():
    data = raw()
    data["last_source_index"] = 3
    data["segments"] = [dict(source_index=i, text="Vâng.", timestamp=ts, speaker=None)
                        for i, ts in enumerate(["10:02", "10:08", "10:05"], 1)]
    assert "TIMESTAMP_ORDER_ERROR" in validate_preserved(data).reason_codes


@pytest.mark.parametrize("indices,reason", [
    ([1, 2, 2, 3], "DUPLICATE_SOURCE_INDEX"),
    ([1, 3, 2, 4], "OUT_OF_ORDER_SOURCE_INDEX"),
    ([1, 2, 4], "MISSING_SOURCE_INDEX"),
    ([1, 2, 3, 4, 5], "OUT_OF_RANGE_SOURCE_INDEX"),
    ([2, 3, 4], "FIRST_SOURCE_INDEX_MISMATCH"),
])
def test_indices_preserved(indices, reason):
    data = raw()
    data["last_source_index"] = 4
    data["segments"] = [dict(source_index=i, text="Vâng.", timestamp=None, speaker=None) for i in indices]
    result = validate_preserved(data, context(last_source_index=4))
    assert not result.is_valid
    assert reason in result.reason_codes


@pytest.mark.parametrize("status", ["FAILED", "PROCESSING"])
def test_non_success_status_is_rejected(status):
    data = raw()
    data["status"] = status
    assert not validate_preserved(data).is_valid


@pytest.mark.parametrize("field,value", [
    ("status", "PARTIAL"), ("status", "OUTPUT_TRUNCATED"),
    ("first_source_index", "1"), ("job_id", None), ("extra", "model metadata"),
])
def test_schema_rejects_unsupported_metadata(field, value):
    data = raw()
    data[field] = value
    before = deepcopy(data)
    with pytest.raises(SchemaValidationError):
        ResponseParser().parse(json.dumps(data))
    assert data == before


@pytest.mark.parametrize("field,value", [("text", None), ("text", 42), ("speaker", 42), ("source_index", True)])
def test_schema_rejects_segment_types(field, value):
    data = raw()
    data["segments"][0][field] = value
    with pytest.raises(SchemaValidationError):
        ResponseParser().parse(json.dumps(data))


def test_dictionary_entrypoint_cannot_discard_unknown_fields():
    data = raw()
    data["extra"] = "bad"
    before = deepcopy(data)
    assert not TranscriptValidator().validate(data, expected_context=context()).is_valid
    assert data == before

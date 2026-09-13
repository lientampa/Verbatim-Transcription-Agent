"""Output merger module for strictly ordered and idempotent segment merging."""

from typing import Iterable
from src.response_parser import TranscriptSegment, TranscriptionBlockResult


class MergeError(ValueError):
    def __init__(self, reason):
        print(f"[MERGE_ERROR] type={reason}", flush=True)
        super().__init__(reason)


class OutputMerger:
    """TEXT_TIMESTAMP merges by canonical source_index. AUDIO merges in physical block
    order with (backend block_id, local ordinal) identity, preserving model fields.
    """

    def __init__(self, initial_segments: Iterable[TranscriptSegment] | None = None, source_mode: str = "TEXT_TIMESTAMP") -> None:
        self.source_mode = source_mode
        self._audio_segments = {}
        self._segments_by_index: dict[int, TranscriptSegment] = {}
        if initial_segments:
            for seg in initial_segments:
                self.add_segment(seg)

    def add_segment(self, segment: TranscriptSegment, block_id: str | None = None) -> None:
        """Add by canonical text index or by AUDIO block-scoped ordinal."""
        if self.source_mode == "AUDIO":
            key = (block_id, segment.source_index)
            if key in self._audio_segments and self._audio_segments[key] != segment:
                raise MergeError("MERGE_CONFLICT")
            self._audio_segments[key] = segment
        else:
            self._segments_by_index[segment.source_index] = segment

    def merge_block_result(self, block_result: TranscriptionBlockResult) -> None:
        """Merge all segments from a validated TranscriptionBlockResult."""
        for seg in block_result.segments:
            self.add_segment(seg, block_result.block_id)

    def get_merged_segments(self) -> list[TranscriptSegment]:
        """Return AUDIO insertion order or canonical TEXT_TIMESTAMP index order."""
        if self.source_mode == "AUDIO":
            return list(self._audio_segments.values())
        return [self._segments_by_index[idx] for idx in sorted(self._segments_by_index.keys())]

    def total_segments(self) -> int:
        return len(self.get_merged_segments())

    def get_last_source_index(self) -> int | None:
        if self.source_mode == "AUDIO":
            segments = self.get_merged_segments()
            return segments[-1].source_index if segments else None
        if not self._segments_by_index:
            return None
        return max(self._segments_by_index.keys())

    def clear(self) -> None:
        self._segments_by_index.clear()
        self._audio_segments.clear()


    @classmethod
    def from_checkpoint(cls, data, require_complete=False):
        """Reconstruct only the payloads in the durable Stage 5 manifest."""
        if data.schema_version != 2 or not data.source_identity:
            raise MergeError("MERGE_SOURCE_IDENTITY_REQUIRED")
        records, offset, last_index = [], 0, 0
        for metric in data.block_metrics:
            count = metric.get("segment_count", metric["last_source_index"] - last_index)
            records.append(dict(metric=metric, segments=data.confirmed_segments[offset:offset+count]))
            offset += count
            last_index = metric["last_source_index"]
        if offset != len(data.confirmed_segments):
            raise MergeError("MERGE_PAYLOAD_COUNT")
        return cls.from_committed_records(records, data.source_identity, require_complete)

    @classmethod
    def from_committed_records(cls, records, source_identity, require_complete=False):
        from copy import deepcopy
        from src.source_identity import to_us
        from src.output_renderer import parse_timestamp_to_seconds
        print(f"[MERGE_START] committed_blocks={len(records)}", flush=True)
        seen, logical, ordered = {}, {}, []
        for record in records:
            metric, payload = record["metric"], record["segments"]
            if metric.get("fidelity_decision") not in ("ACCEPT", "ACCEPT_WITH_WARNING") or metric.get("coverage", {}).get("decision") not in ("COVERAGE_PASS", "COVERAGE_WARNING"):
                raise MergeError("MERGE_UNCOMMITTED_OR_REJECTED")
            if not payload or metric.get("segment_count", len(payload)) != len(payload):
                raise MergeError("MERGE_PAYLOAD_COUNT")
            start, end = to_us(metric["actual_start_offset"]), to_us(metric["actual_end_offset"])
            block_id = metric["block_id"]
            key = (source_identity["fingerprint"], start, end, block_id)
            if block_id in logical and logical[block_id] != key:
                raise MergeError("MERGE_MULTIPLE_PHYSICAL_GENERATIONS")
            logical[block_id] = key
            if key in seen:
                if seen[key] != payload:
                    raise MergeError("MERGE_CONFLICT")
                print(f"[MERGE_DUPLICATE_IDENTICAL] block_id={block_id}")
                continue
            seen[key] = deepcopy(payload)
            ordered.append((start, end, block_id, payload))
        result = cls(source_mode="AUDIO")
        previous_end, previous_timestamp = 0, -1
        for number, (start, end, block_id, payload) in enumerate(sorted(ordered, key=lambda r: r[0]), 1):
            if start > previous_end:
                raise MergeError("MERGE_GAP")
            if start < previous_end:
                raise MergeError("MERGE_OVERLAP")
            if block_id != f"BLOCK_{number:03d}":
                raise MergeError("MERGE_LOGICAL_PHYSICAL_ORDER_CONFLICT")
            if not start < end <= source_identity["duration_us"]:
                raise MergeError("MERGE_RANGE_INVALID")
            previous_ordinal = 0
            for item in payload:
                segment = TranscriptSegment.from_dict(deepcopy(item))
                timestamp = parse_timestamp_to_seconds(segment.timestamp)
                if timestamp is None or timestamp < previous_timestamp:
                    raise MergeError("MERGE_TIMESTAMP_ORDER")
                # Match the existing upstream one-second boundary tolerance.
                if not start / 1_000_000 - 1 <= timestamp <= min(end / 1_000_000 + 1, source_identity["duration_us"] / 1_000_000):
                    raise MergeError("MERGE_TIMESTAMP_OUT_OF_RANGE")
                if segment.source_index <= previous_ordinal:
                    raise MergeError("MERGE_SEGMENT_ORDER")
                result.add_segment(segment, block_id)
                previous_timestamp, previous_ordinal = timestamp, segment.source_index
            previous_end = end
        if require_complete and previous_end != source_identity["duration_us"]:
            raise MergeError("MERGE_SOURCE_INCOMPLETE")
        print(f"[MERGE_VALIDATE] status=PASS physical_start=0 physical_end={previous_end/1_000_000} segments={result.total_segments()}", flush=True)
        return result

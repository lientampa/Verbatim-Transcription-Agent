"""Output merger module for strictly ordered and idempotent segment merging."""

from typing import Iterable
from src.response_parser import TranscriptSegment, TranscriptionBlockResult


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
            self._audio_segments[(block_id, segment.source_index)] = segment
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

"""Output merger module for strictly ordered and idempotent segment merging."""

from typing import Iterable
from src.response_parser import TranscriptSegment, TranscriptionBlockResult


class OutputMerger:
    """Merges transcription segments across blocks strictly ordered by source_index.

    Ensures idempotency: duplicate segments with the same source_index are deduplicated.
    """

    def __init__(self, initial_segments: Iterable[TranscriptSegment] | None = None) -> None:
        self._segments_by_index: dict[int, TranscriptSegment] = {}
        if initial_segments:
            for seg in initial_segments:
                self.add_segment(seg)

    def add_segment(self, segment: TranscriptSegment) -> None:
        """Add or update segment idempotently by source_index."""
        self._segments_by_index[segment.source_index] = segment

    def merge_block_result(self, block_result: TranscriptionBlockResult) -> None:
        """Merge all segments from a validated TranscriptionBlockResult."""
        for seg in block_result.segments:
            self.add_segment(seg)

    def get_merged_segments(self) -> list[TranscriptSegment]:
        """Return all merged segments sorted strictly ascending by source_index."""
        return [self._segments_by_index[idx] for idx in sorted(self._segments_by_index.keys())]

    def total_segments(self) -> int:
        return len(self._segments_by_index)

    def get_last_source_index(self) -> int | None:
        if not self._segments_by_index:
            return None
        return max(self._segments_by_index.keys())

    def clear(self) -> None:
        self._segments_by_index.clear()

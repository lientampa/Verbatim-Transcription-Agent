"""Output renderer generating transcript.txt, transcript.docx, and subtitle.srt from validated structured segments."""

from pathlib import Path
from typing import Sequence
import os
import re
import tempfile
import math
import json
import shutil
import docx
from docx.shared import Pt, RGBColor
from src.response_parser import TranscriptSegment


def parse_timestamp_to_seconds(ts: str | None) -> float | None:
    """Parse 'HH:MM:SS' or 'MM:SS' string to float seconds. Return None if invalid or None."""
    if not ts:
        return None
    cleaned = ts.strip().strip("[]")
    parts = cleaned.split(":")
    try:
        if len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
            return h * 3600 + m * 60 + s if h >= 0 and 0 <= m < 60 and 0 <= s < 60 and math.isfinite(s) else None
        elif len(parts) == 2:
            m, s = int(parts[0]), float(parts[1])
            return m * 60 + s if m >= 0 and 0 <= s < 60 and math.isfinite(s) else None
    except (ValueError, TypeError):
        return None
    return None


def format_seconds_to_srt_time(seconds: float) -> str:
    """Convert float seconds to SRT time string 'HH:MM:SS,mmm'."""
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3_600_000
    remainder = total_ms % 3_600_000
    minutes = remainder // 60_000
    remainder = remainder % 60_000
    secs = remainder // 1000
    millis = remainder % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def normalize_timestamp_display(ts: str | None) -> str | None:
    """Ensure timestamp displays cleanly as HH:MM:SS or MM:SS."""
    if not ts:
        return None
    cleaned = ts.strip().strip("[]")
    parts = cleaned.split(":")
    if len(parts) == 2:
        try:
            m, s = int(parts[0]), int(parts[1])
            return f"{m:02d}:{s:02d}"
        except ValueError:
            pass
    elif len(parts) == 3:
        try:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
            return f"{h:02d}:{m:02d}:{s:02d}"
        except ValueError:
            pass
    return cleaned


class OutputRenderer:
    """Renders validated transcription segments into TXT, DOCX, and SRT formats."""

    def __init__(self, default_segment_duration_seconds: float = 4.0, strict_speaker_format: bool = False, srt_end_policy: str = "NEXT_START", srt_min_duration: float = 0.1, source_end: float | None = None) -> None:
        self.default_segment_duration_seconds = default_segment_duration_seconds
        self.strict_speaker_format = strict_speaker_format
        if srt_end_policy not in ("NEXT_START", "ESTIMATED"):
            raise ValueError("SRT_END_POLICY_UNSUPPORTED: schema has no model end")
        if not math.isfinite(srt_min_duration) or srt_min_duration < .001 or not math.isfinite(default_segment_duration_seconds) or default_segment_duration_seconds <= 0:
            raise ValueError("SRT_DURATION_INVALID")
        self.srt_end_policy = srt_end_policy
        self.srt_min_duration = srt_min_duration
        self.source_end = source_end
        self.render_contract_version = 2

    def validate_output_fields(self, segments):
        if not self.strict_speaker_format:
            return
        for segment in segments:
            label = segment.speaker or ""
            if (not label.strip() or re.search(r"[\r\n\[\]:=]", label)
                    or re.fullmatch(r"(?:speaker|segment|block|unknown)[_ -]*[a-z0-9]+", label.strip(), re.I)):
                raise ValueError(f"SPEAKER_LABEL_REQUIRED: segment ordinal {segment.source_index}; human-readable stable label required")
            if "\n" in segment.text or "\r" in segment.text:
                raise ValueError("RENDER_MULTILINE_SEGMENT")
            if parse_timestamp_to_seconds(segment.timestamp) is None or not segment.timestamp or not re.fullmatch(r"(?:[0-9]{1,2}:)?[0-9]{1,2}:[0-9]{2}", segment.timestamp):
                raise ValueError(f"ABSOLUTE_TIMESTAMP_REQUIRED: segment ordinal {segment.source_index}")

    def _display_timestamp(self, segment):
        self.validate_output_fields([segment])
        if self.strict_speaker_format:
            return format_seconds_to_srt_time(parse_timestamp_to_seconds(segment.timestamp)).split(",")[0]
        return normalize_timestamp_display(segment.timestamp)


    def format_txt_line(self, segment: TranscriptSegment) -> str:
        """Format a single segment for transcript.txt.

        Format:
        - [HH:MM:SS - Speaker]: Text
        - [Speaker]: Text (if no timestamp)
        - [HH:MM:SS]: Text (if no speaker)
        - Text (if neither)
        """
        ts_str = self._display_timestamp(segment)
        speaker_str = segment.speaker.strip() if segment.speaker and segment.speaker.strip() else None

        if ts_str and speaker_str:
            header = f"[{ts_str} - {speaker_str}]: "
        elif speaker_str:
            header = f"[{speaker_str}]: "
        elif ts_str:
            header = f"[{ts_str}]: "
        else:
            header = ""

        return f"{header}{segment.text}"

    def render_txt(self, segments: Sequence[TranscriptSegment], output_path: Path) -> Path:
        """Render formatted transcript to plain text file with blank line separators."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [self.format_txt_line(seg) for seg in segments]
        content = ("\n" if self.strict_speaker_format else "\n\n").join(lines)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
            if content:
                f.write("\n")

        return output_path

    def render_docx(self, segments: Sequence[TranscriptSegment], output_path: Path, title: str = "BẢN PHIÊN ÂM NGUYÊN VĂN") -> Path:
        """Render formatted transcript into a Microsoft Word (.docx) document.

        Uses write-to-temp-then-replace strategy to avoid PermissionError when
        the target .docx file is currently open in Microsoft Word.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        doc = docx.Document()

        # Add Title
        h = doc.add_heading(title, level=1)
        h.paragraph_format.space_after = Pt(14)

        for seg in segments:
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(8)
            p.paragraph_format.line_spacing = 1.15

            ts_str = self._display_timestamp(seg)
            speaker_str = seg.speaker.strip() if seg.speaker and seg.speaker.strip() else None

            if ts_str and speaker_str:
                prefix = f"[{ts_str} - {speaker_str}]: "
            elif speaker_str:
                prefix = f"[{speaker_str}]: "
            elif ts_str:
                prefix = f"[{ts_str}]: "
            else:
                prefix = ""

            if prefix:
                run_prefix = p.add_run(prefix)
                run_prefix.bold = True
                run_prefix.font.color.rgb = RGBColor(0, 51, 102)

            run_text = p.add_run(seg.text)
            run_text.font.size = Pt(11)

        # Write to temp file in same directory, then atomically replace target.
        # This avoids PermissionError when Word has the target file open.
        tmp_path: Path | None = None
        try:
            fd, tmp_str = tempfile.mkstemp(
                suffix=".docx",
                dir=output_path.parent,
                prefix="_tmp_transcript_",
            )
            os.close(fd)
            tmp_path = Path(tmp_str)
            doc.save(str(tmp_path))
            # os.replace is atomic on Windows and overwrites locked files if possible
            os.replace(str(tmp_path), str(output_path))
            tmp_path = None  # Successfully replaced — no cleanup needed
        except PermissionError as exc:
            raise PermissionError(
                f"Không thể ghi file DOCX '{output_path.name}'. "
                f"Hãy đóng file này trong Microsoft Word rồi chạy lại.\n(Chi tiết: {exc})"
            ) from exc
        finally:
            # Clean up temp file if replace failed
            if tmp_path and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

        return output_path

    def render_srt(self, segments: Sequence[TranscriptSegment], output_path: Path) -> Path:
        """Render validated segments to standard SubRip (.srt) subtitle format."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        srt_blocks = []
        for i, (start_sec, end_sec) in enumerate(self.srt_times(segments), 1):
            seg = segments[i-1]
            # Retain the existing explicit product convention in strict mode:
            # the user previously required the bracketed line in all formats.
            speaker_tag = f"[{seg.speaker.strip()}]: " if seg.speaker and seg.speaker.strip() else ""
            sub_text = self.format_txt_line(seg) if self.strict_speaker_format else f"{speaker_tag}{seg.text}"
            srt_blocks.append(f"{i}\n{format_seconds_to_srt_time(start_sec)} --> {format_seconds_to_srt_time(end_sec)}\n{sub_text}")

        content = "\n\n".join(srt_blocks)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
            if content:
                f.write("\n")

        return output_path

    def srt_times(self, segments):
        starts = [parse_timestamp_to_seconds(seg.timestamp) for seg in segments]
        if any(t is None for t in starts):
            if self.strict_speaker_format:
                raise ValueError("SRT_START_REQUIRED")
            starts = [t if t is not None else i*self.default_segment_duration_seconds for i,t in enumerate(starts)]
        if any(b < a and not (getattr(segments[i],"_native_text_order_block",None) and
                              getattr(segments[i],"_native_text_order_block",None)==getattr(segments[i+1],"_native_text_order_block",None))
               for i,(a,b) in enumerate(zip(starts, starts[1:]))):
            raise ValueError("SRT_START_ORDER")
        times = []
        for i, start in enumerate(starts):
            end = (starts[i+1] if self.srt_end_policy == "NEXT_START" and i+1 < len(starts)
                   else start + self.default_segment_duration_seconds)
            end = max(end, start + self.srt_min_duration)
            if self.source_end is not None:
                end = min(end, self.source_end)
            if round(end*1000) <= round(start*1000):
                raise ValueError("SRT_NONPOSITIVE_DURATION")
            times.append((start,end))
        return times

    def render_all(self, segments, base_dir, job_id=None):
        """Stage and validate the complete set before publishing, with rollback."""
        from src.renderer_validator import RendererValidator
        from src.render_transaction import publish_outputs, recover_outputs
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        recover_outputs(base_dir)
        self.validate_output_fields(segments)
        with tempfile.TemporaryDirectory(prefix=".render-stage-", dir=base_dir) as temporary:
            stage = Path(temporary)
            self.render_txt(segments, stage / "transcript.txt")
            self.render_docx(segments, stage / "transcript.docx")
            self.render_srt(segments, stage / "subtitle.srt")
            RendererValidator.validate(self, segments, stage)
            (stage / "render_manifest.json").write_text(json.dumps({
                "render_contract_version": self.render_contract_version,
                "segment_count": len(segments), "srt_end_policy": self.srt_end_policy,
                "srt_end_is_estimated": True, "source_end": self.source_end, **getattr(self, "provenance", {})}), encoding="utf-8")
            publish_outputs(stage, base_dir)
        return {"txt": base_dir / "transcript.txt", "docx": base_dir / "transcript.docx", "srt": base_dir / "subtitle.srt"}

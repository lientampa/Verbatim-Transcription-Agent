"""Output renderer generating transcript.txt, transcript.docx, and subtitle.srt from validated structured segments."""

from pathlib import Path
from typing import Sequence
import os
import tempfile
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
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = int(parts[0]), float(parts[1])
            return m * 60 + s
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

    def __init__(self, default_segment_duration_seconds: float = 4.0) -> None:
        self.default_segment_duration_seconds = default_segment_duration_seconds

    def format_txt_line(self, segment: TranscriptSegment) -> str:
        """Format a single segment for transcript.txt.

        Format:
        - [HH:MM:SS - Speaker]: Text
        - [Speaker]: Text (if no timestamp)
        - [HH:MM:SS]: Text (if no speaker)
        - Text (if neither)
        """
        ts_str = normalize_timestamp_display(segment.timestamp)
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
        content = "\n\n".join(lines)

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

            ts_str = normalize_timestamp_display(segment_ts := seg.timestamp)
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

        srt_blocks: list[str] = []
        cue_index = 1

        total_segments = len(segments)
        for i, seg in enumerate(segments):
            start_sec = parse_timestamp_to_seconds(seg.timestamp)
            if start_sec is None:
                # If no timestamp, calculate sequential offset based on cue index
                start_sec = (cue_index - 1) * self.default_segment_duration_seconds

            # Explicit policy for end timestamp calculation without hallucination:
            # If next segment has valid timestamp > start_sec, use min(next_start, start_sec + default_duration)
            end_sec: float
            if i + 1 < total_segments:
                next_start = parse_timestamp_to_seconds(segments[i + 1].timestamp)
                if next_start is not None and next_start > start_sec:
                    end_sec = min(next_start, start_sec + self.default_segment_duration_seconds)
                else:
                    end_sec = start_sec + self.default_segment_duration_seconds
            else:
                end_sec = start_sec + self.default_segment_duration_seconds

            start_srt = format_seconds_to_srt_time(start_sec)
            end_srt = format_seconds_to_srt_time(end_sec)

            speaker_tag = f"[{seg.speaker.strip()}]: " if seg.speaker and seg.speaker.strip() else ""
            sub_text = f"{speaker_tag}{seg.text}"

            cue = f"{cue_index}\n{start_srt} --> {end_srt}\n{sub_text}"
            srt_blocks.append(cue)
            cue_index += 1

        content = "\n\n".join(srt_blocks)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
            if content:
                f.write("\n")

        return output_path

    def render_all(
        self,
        segments: Sequence[TranscriptSegment],
        base_dir: Path,
        job_id: str | None = None,
    ) -> dict[str, Path]:
        """Render all 3 required output formats: TXT, DOCX, and SRT.

        Each format is rendered independently. A PermissionError on DOCX
        (e.g. file open in Word) raises immediately with a clear message.
        TXT and SRT are always attempted.
        """
        base_dir = Path(base_dir)
        txt_path = base_dir / "transcript.txt"
        docx_path = base_dir / "transcript.docx"
        srt_path = base_dir / "subtitle.srt"

        self.render_txt(segments, txt_path)
        self.render_srt(segments, srt_path)
        # DOCX last — most likely to be locked. TXT & SRT already saved.
        self.render_docx(segments, docx_path)

        return {
            "txt": txt_path,
            "docx": docx_path,
            "srt": srt_path,
        }

"""Read-back parity against canonical segments; no content repair."""
import docx
import re
from src.output_renderer import format_seconds_to_srt_time


class RendererValidator:
    @staticmethod
    def validate(renderer, segments, directory):
        expected = [renderer.format_txt_line(seg) for seg in segments]
        actual = (directory / "transcript.txt").read_text(encoding="utf-8")
        separator = "\n" if renderer.strict_speaker_format else "\n\n"
        if actual != separator.join(expected) + ("\n" if expected else ""):
            raise ValueError("RENDER_TXT_PARITY")
        paragraphs = [p.text for p in docx.Document(directory / "transcript.docx").paragraphs][1:]
        if paragraphs != expected:
            raise ValueError("RENDER_DOCX_PARITY")
        cues = []
        for index, (seg, (start,end)) in enumerate(zip(segments, renderer.srt_times(segments)), 1):
            text = renderer.format_txt_line(seg) if renderer.strict_speaker_format else ((f"[{seg.speaker.strip()}]: " if seg.speaker and seg.speaker.strip() else "") + seg.text)
            cues.append(f"{index}\n{format_seconds_to_srt_time(start)} --> {format_seconds_to_srt_time(end)}\n{text}")
        if (directory / "subtitle.srt").read_text(encoding="utf-8") != "\n\n".join(cues) + ("\n" if cues else ""):
            raise ValueError("RENDER_SRT_PARITY")
        raw_cues = (directory / "subtitle.srt").read_text(encoding="utf-8").strip().split("\n\n") if segments else []
        if len(raw_cues) != len(segments):
            raise ValueError("RENDER_SRT_COUNT")
        previous_start = -1
        for number, cue in enumerate(raw_cues, 1):
            lines = cue.splitlines()
            if len(lines) < 3 or lines[0] != str(number):
                raise ValueError("RENDER_SRT_NUMBERING")
            match = re.fullmatch(r"(\d{2,}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2,}):(\d{2}):(\d{2}),(\d{3})", lines[1])
            if not match:
                raise ValueError("RENDER_SRT_TIMING_FORMAT")
            h,m,s,ms,eh,em,es,ems = map(int, match.groups())
            start = ((h*60+m)*60+s)*1000+ms
            end = ((eh*60+em)*60+es)*1000+ems
            native_pair=(number>1 and getattr(segments[number-1],"_native_text_order_block",None) and
                         getattr(segments[number-1],"_native_text_order_block",None)==getattr(segments[number-2],"_native_text_order_block",None))
            if max(m,s,em,es) >= 60 or (start < previous_start and not native_pair) or end <= start:
                raise ValueError("RENDER_SRT_TIMING_INVALID")
            previous_start = start
        print(f"[RENDER_TXT] segments={len(segments)} status=PASS")
        print(f"[RENDER_DOCX] segments={len(segments)} status=PASS")
        print(f"[RENDER_SRT] cues={len(segments)} policy={renderer.srt_end_policy} status=PASS")
        print("[FINAL_OUTPUT_VALIDATE] status=PASS")

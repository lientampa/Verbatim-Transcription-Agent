"""Audio file discovery and validation manager."""

from pathlib import Path
from dataclasses import dataclass


class AudioError(Exception):
    """Base exception for audio manager errors."""
    pass


class AudioNotFoundError(AudioError):
    """Raised when no audio file is found in the target directory."""
    pass


class UnsupportedAudioFormatError(AudioError):
    """Raised when an audio file has an unsupported format."""
    pass


@dataclass(frozen=True)
class AudioFileInfo:
    """Information about a discovered audio file."""
    path: Path
    file_name: str
    extension: str
    size_bytes: int


class AudioManager:
    """Manages audio file scanning, validation, and preparation."""

    SUPPORTED_EXTENSIONS: set[str] = {
        ".mp3",
        ".m4a",
        ".wav",
        ".mp4",
        ".aac",
        ".flac",
        ".ogg",
    }

    def __init__(self, audio_dir: Path) -> None:
        self.audio_dir = Path(audio_dir)

    def find_target_audio(self) -> AudioFileInfo:
        """Find and validate a single target audio file in the audio directory.

        In MVP 1, only one audio file is processed per session.
        If multiple audio files exist, the first valid audio file is selected
        in deterministic sorted order.

        Returns:
            AudioFileInfo describing the audio file.

        Raises:
            AudioNotFoundError: If the directory has no valid audio files.
            UnsupportedAudioFormatError: If files exist but none are supported.
        """
        if not self.audio_dir.exists():
            raise AudioNotFoundError(f"Audio directory does not exist: {self.audio_dir}")

        all_files = [
            f for f in sorted(self.audio_dir.iterdir())
            if f.is_file() and not f.name.startswith(".")
        ]

        if not all_files:
            raise AudioNotFoundError(
                f"No audio file found in '{self.audio_dir}'. "
                f"Please place an audio file (e.g. sample.mp3) into the audio directory."
            )

        supported_files: list[Path] = []
        unsupported_files: list[Path] = []

        for file_path in all_files:
            ext = file_path.suffix.lower()
            if ext in self.SUPPORTED_EXTENSIONS:
                supported_files.append(file_path)
            else:
                unsupported_files.append(file_path)

        if not supported_files:
            invalid_list = ", ".join(f.name for f in unsupported_files)
            supported_list = ", ".join(sorted(self.SUPPORTED_EXTENSIONS))
            raise UnsupportedAudioFormatError(
                f"Found files [{invalid_list}] in '{self.audio_dir}', but none have supported extensions. "
                f"Supported formats are: {supported_list}"
            )

        target_file = supported_files[0]
        size_bytes = target_file.stat().st_size

        if size_bytes == 0:
            raise AudioError(f"Audio file is empty (0 bytes): {target_file.name}")

        return AudioFileInfo(
            path=target_file,
            file_name=target_file.name,
            extension=target_file.suffix.lower(),
            size_bytes=size_bytes,
        )

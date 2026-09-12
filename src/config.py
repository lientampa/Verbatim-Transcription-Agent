"""Configuration management for Vietnamese Verbatim Transcription Agent."""

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""
    pass


@dataclass(frozen=True)
class AppConfig:
    """Application configuration holder."""

    # API configuration
    gemini_api_key: str
    gemini_model: str

    # Paths
    base_dir: Path
    audio_dir: Path
    output_dir: Path
    state_dir: Path
    prompts_dir: Path
    system_prompt_path: Path
    output_transcript_path: Path
    output_docx_path: Path
    output_srt_path: Path
    checkpoint_file_path: Path
    schema_path: Path

    # Retry and timeout parameters
    retry_max_attempts: int
    retry_initial_delay_seconds: float
    timeout_seconds: int

    # Milestone 2: Block Builder parameters
    block_duration_seconds: int
    cache_dir: Path

    # Milestone 3: Validator parameters
    validator_max_retries: int

    # Dynamic Adaptive Block Builder (DABB) configuration
    model_context_limit: int = 1_000_000
    model_max_output_tokens: int = 8192
    context_safety_margin: int = 2000
    output_safety_margin: int = 1000
    initial_target_input_tokens: int = 2000
    min_block_tokens: int = 300
    max_block_tokens: int = 6000
    block_growth_step: int = 500
    block_shrink_factor: float = 0.6
    default_output_ratio: float = 1.2
    min_duration_seconds: float = 60.0
    target_duration_seconds: float = 300.0
    max_duration_seconds: float = 900.0



def load_config(base_dir: Path | None = None) -> AppConfig:
    """Load configuration from environment variables and .env file.

    Args:
        base_dir: Optional root directory path. Defaults to project root.

    Returns:
        AppConfig instance.

    Raises:
        ConfigurationError: If required configuration is missing or invalid.
    """
    if base_dir is None:
        # Resolve to root directory containing src/
        base_dir = Path(__file__).resolve().parent.parent
        env_path = base_dir / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
    else:
        env_path = base_dir / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        raise ConfigurationError(
            "GEMINI_API_KEY is not set or contains placeholder value. "
            "Please create or update your .env file with a valid Gemini API key. "
            "See .env.example for reference."
        )

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
    if not model_name:
        model_name = "gemini-2.5-flash"

    try:
        retry_max_attempts = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
    except ValueError:
        retry_max_attempts = 3

    try:
        retry_initial_delay_seconds = float(os.getenv("RETRY_INITIAL_DELAY_SECONDS", "2.0"))
    except ValueError:
        retry_initial_delay_seconds = 2.0

    try:
        timeout_seconds = int(os.getenv("TIMEOUT_SECONDS", "300"))
    except ValueError:
        timeout_seconds = 300

    try:
        block_duration_seconds = int(os.getenv("BLOCK_DURATION_SECONDS", "300"))
    except ValueError:
        block_duration_seconds = 300

    try:
        validator_max_retries = int(os.getenv("VALIDATOR_MAX_RETRIES", "2"))
    except ValueError:
        validator_max_retries = 2

    # DABB Parameters
    model_context_limit = int(os.getenv("MODEL_CONTEXT_LIMIT", "1000000"))
    model_max_output_tokens = int(os.getenv("MODEL_MAX_OUTPUT_TOKENS", "8192"))
    context_safety_margin = int(os.getenv("CONTEXT_SAFETY_MARGIN", "2000"))
    output_safety_margin = int(os.getenv("OUTPUT_SAFETY_MARGIN", "1000"))
    initial_target_input_tokens = int(os.getenv("INITIAL_TARGET_INPUT_TOKENS", "2000"))
    min_block_tokens = int(os.getenv("MIN_BLOCK_TOKENS", "300"))
    max_block_tokens = int(os.getenv("MAX_BLOCK_TOKENS", "6000"))
    block_growth_step = int(os.getenv("BLOCK_GROWTH_STEP", "500"))
    block_shrink_factor = float(os.getenv("BLOCK_SHRINK_FACTOR", "0.6"))
    default_output_ratio = float(os.getenv("DEFAULT_OUTPUT_RATIO", "1.2"))
    min_duration_seconds = float(os.getenv("MIN_DURATION", "60.0"))
    target_duration_seconds = float(os.getenv("TARGET_DURATION", "300.0"))
    max_duration_seconds = float(os.getenv("MAX_DURATION", "900.0"))

    audio_dir = base_dir / "audio"
    output_dir = base_dir / "output"
    state_dir = base_dir / "state"
    prompts_dir = base_dir / "prompts"
    cache_dir = base_dir / ".cache" / "blocks"

    # Ensure required directories exist
    audio_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    return AppConfig(
        gemini_api_key=api_key,
        gemini_model=model_name,
        base_dir=base_dir,
        audio_dir=audio_dir,
        output_dir=output_dir,
        state_dir=state_dir,
        prompts_dir=prompts_dir,
        system_prompt_path=prompts_dir / "system_prompt.txt",
        output_transcript_path=output_dir / "transcript.txt",
        output_docx_path=output_dir / "transcript.docx",
        output_srt_path=output_dir / "subtitle.srt",
        checkpoint_file_path=state_dir / "checkpoint.json",
        schema_path=base_dir / "schemas" / "transcription_result.schema.json",
        retry_max_attempts=max(1, retry_max_attempts),
        retry_initial_delay_seconds=max(0.1, retry_initial_delay_seconds),
        timeout_seconds=max(10, timeout_seconds),
        block_duration_seconds=max(30, block_duration_seconds),
        cache_dir=cache_dir,
        validator_max_retries=max(1, validator_max_retries),
        model_context_limit=model_context_limit,
        model_max_output_tokens=model_max_output_tokens,
        context_safety_margin=context_safety_margin,
        output_safety_margin=output_safety_margin,
        initial_target_input_tokens=initial_target_input_tokens,
        min_block_tokens=min_block_tokens,
        max_block_tokens=max_block_tokens,
        block_growth_step=block_growth_step,
        block_shrink_factor=block_shrink_factor,
        default_output_ratio=default_output_ratio,
        min_duration_seconds=min_duration_seconds,
        target_duration_seconds=target_duration_seconds,
        max_duration_seconds=max_duration_seconds,
    )

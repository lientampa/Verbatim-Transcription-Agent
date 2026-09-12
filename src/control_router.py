"""Control Router Interface (Prepared for Future Milestones).

Separates CONTROL INPUT (system commands: START, RESUME, STOP, REPROCESS, NEW_SESSION)
from SOURCE INPUT (raw spoken audio content).
Ensures control directives are never conflated with transcript content.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class ControlCommand(str, Enum):
    """Supported control commands in the transcription lifecycle."""
    START = "START"
    RESUME = "RESUME"
    TIEP_TUC = "TIẾP TỤC"
    STOP = "STOP"
    RETRY = "RETRY"
    RESET = "RESET"
    NEW_SESSION = "NEW_SESSION"


CONTROL_COMMAND_ALIASES: dict[str, ControlCommand] = {
    "start": ControlCommand.START,
    "resume": ControlCommand.RESUME,
    "tiếp tục": ControlCommand.TIEP_TUC,
    "tiep tuc": ControlCommand.TIEP_TUC,
    "stop": ControlCommand.STOP,
    "retry": ControlCommand.RETRY,
    "reset": ControlCommand.RESET,
    "new_session": ControlCommand.NEW_SESSION,
    "new session": ControlCommand.NEW_SESSION,
}


def parse_control_command(text: str) -> ControlCommand | None:
    """Check if an input string is a system control command.

    Used at the application/control layer to ensure control directives
    are routed properly and NEVER conflated with transcript content.
    """
    cleaned = text.strip().lower()
    return CONTROL_COMMAND_ALIASES.get(cleaned)


class ControlRouter(ABC):
    """Abstract interface for routing user and system lifecycle commands."""

    @abstractmethod
    def route_command(self, command: ControlCommand, context: dict[str, Any] | None = None) -> Any:
        """Route and execute a control command without polluting source transcript.

        Args:
            command: The lifecycle control directive.
            context: Optional contextual parameters (e.g. job_id, resume_point).
        """
        pass


class StandardControlRouter(ControlRouter):
    """Concrete control router managing system lifecycle operations."""

    def route_command(self, command: ControlCommand, context: dict[str, Any] | None = None) -> dict[str, Any]:
        context = context or {}
        cmd_str = command.value if isinstance(command, ControlCommand) else str(command)

        return {
            "status": "PROCESSED",
            "command": cmd_str,
            "handled_at": "control_layer",
            "context": context,
        }


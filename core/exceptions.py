"""Custom exception types used across the project.

Keeping these centralized makes it easy for the agent loop to catch
specific, expected failure modes (bad JSON, unknown tool, bad arguments)
without accidentally swallowing real bugs.
"""


class ProjectError(Exception):
    """Base class for all project-specific exceptions."""


class SourceConfigError(ProjectError):
    """Raised when a source configuration file is missing or invalid."""


class ToolNotFoundError(ProjectError):
    """Raised when the model requests a tool that is not registered."""


class ToolDisabledError(ProjectError):
    """Raised when the model requests a tool that exists but is disabled."""


class InvalidArgumentsError(ProjectError):
    """Raised when arguments supplied to a tool fail schema validation."""


class ModelOutputParseError(ProjectError):
    """Raised when the model's raw output cannot be parsed as the expected
    JSON action contract.
    """


class ToolExecutionError(ProjectError):
    """Raised when a tool's underlying operation fails at runtime.

    This is distinct from InvalidArgumentsError: the arguments were valid,
    but the operation itself failed (e.g. network error, file not found).
    """

    def __init__(self, message: str, error_type: str = "ToolExecutionError", recoverable: bool = True):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.recoverable = recoverable

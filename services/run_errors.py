class RunExecutionError(RuntimeError):
    """Base class for errors that belong to run execution."""


class TransientRunError(RunExecutionError):
    """A temporary failure that can be retried from a checkpoint."""


class PermanentRunError(RunExecutionError):
    """A failure that should terminalize the run instead of retrying."""


class ApprovalRequired(RunExecutionError):
    """Execution must pause until a human supplies an approval or input."""


class RunCancelled(RunExecutionError):
    """Execution reached a safe point after cancellation was requested."""


class RetryableRunError(TransientRunError):
    """Backward-compatible name for the old worker retry protocol."""

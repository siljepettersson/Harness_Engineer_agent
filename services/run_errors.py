class RetryableRunError(RuntimeError):
    """A dependency failure that should return the run to the queue."""

class RunExecutor:
    """Application boundary for the Harness Loop implementation."""

    def execute(self, run_id: str, worker_id: str) -> dict[str, str]:
        raise NotImplementedError(
            "The Harness Loop executor has not been implemented yet"
        )


run_executor = RunExecutor()


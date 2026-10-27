from copy import deepcopy
from typing import Any, Iterable

from services.run_errors import PermanentRunError
from services.run_executor import ModelDecision


class ScriptedModelProvider:
    """Deterministic provider for executor tests and local demonstrations."""

    def __init__(self, responses: Iterable[ModelDecision]) -> None:
        self._responses = list(responses)
        self.contexts: list[dict[str, Any]] = []

    def complete(self, *, context: dict[str, Any]) -> ModelDecision:
        self.contexts.append(deepcopy(context))
        if not self._responses:
            raise PermanentRunError("scripted_model_exhausted")
        return self._responses.pop(0)

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from repositories.run_repository import SqlAlchemyRunRepository, run_repository
from schemas import (
    CancelRunRequest,
    CreateRunRequest,
    ResumeRunRequest,
    RerunRunRequest,
    RunResponse,
)
from state_machine import InvalidRunTransition


class ServiceError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class RunService:
    def __init__(self, repository: SqlAlchemyRunRepository) -> None:
        self.repository = repository

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _event(
        run_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "id": str(uuid4()),
            "run_id": run_id,
            "sequence": 0,
            "event_type": event_type,
            "payload": payload,
            "created_at": RunService._now(),
        }

    @staticmethod
    def _input_hash(input_data: dict[str, Any]) -> str:
        import hashlib
        import json

        canonical = json.dumps(
            input_data,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _to_response(run: dict[str, Any]) -> RunResponse:
        return RunResponse(
            run_id=run["id"],
            org_id=run["org_id"],
            user_id=run["user_id"],
            parent_run_id=run.get("parent_run_id"),
            run_type=run["run_type"],
            status=run["status"],
            input=run["input"],
            output=run["output"],
            error=run["error"],
            created_at=run["created_at"],
            updated_at=run["updated_at"],
        )

    @staticmethod
    def _not_found() -> ServiceError:
        return ServiceError(404, "RUN_NOT_FOUND", "Run not found")

    def create_run(
        self,
        org_id: str,
        user_id: str,
        request: CreateRunRequest,
    ) -> tuple[RunResponse, bool]:
        now = self._now()
        run_id = str(uuid4())
        input_hash = self._input_hash(request.input)
        run = {
            "id": run_id,
            "org_id": org_id,
            "user_id": user_id,
            "run_type": request.run_type,
            "idempotency_key": request.idempotency_key,
            "input": request.input,
            "input_hash": input_hash,
            "status": "queued",
            "output": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
        }
        event = self._event(
            run_id,
            "run.created",
            {"status": "queued", "run_type": request.run_type},
        )
        event["sequence"] = 1

        stored_run, created = self.repository.create_if_absent(
            org_id,
            run,
            event,
        )

        if not created:
            same_input = stored_run.get("input") == request.input
            stored_hash = stored_run.get("input_hash")
            if (stored_hash is not None and stored_hash != input_hash) or not same_input:
                raise ServiceError(
                    409,
                    "IDEMPOTENCY_CONFLICT",
                    "Idempotency key already used with a different input",
                )
            return self._to_response(stored_run), False

        return self._to_response(stored_run), True

    def _get_authorized_run(
        self,
        run_id: str,
        org_id: str,
    ) -> dict[str, Any]:
        run = self.repository.get(run_id, org_id)
        if run is None:
            raise self._not_found()
        return run

    def get_run(self, run_id: str, org_id: str) -> RunResponse:
        return self._to_response(self._get_authorized_run(run_id, org_id))

    def get_events(self, run_id: str, org_id: str) -> list[dict[str, Any]]:
        self._get_authorized_run(run_id, org_id)
        return self.repository.list_events(run_id, org_id)

    def cancel_run(
        self,
        run_id: str,
        org_id: str,
        user_id: str,
        request: CancelRunRequest,
    ) -> RunResponse:
        run = self._get_authorized_run(run_id, org_id)
        # Preserve the API's established terminal-cancel contract while the
        # repository still enforces InvalidRunTransition for all transitions.
        if run["status"] in {"succeeded", "failed", "cancelled"}:
            raise ServiceError(
                409,
                "RUN_ALREADY_TERMINAL",
                "Run is already terminal",
            )
        try:
            updated_run = self.repository.request_cancel(
                run_id=run_id,
                org_id=org_id,
                actor_type="user",
                actor_id=user_id,
                reason=request.reason,
            )
        except KeyError:
            raise self._not_found()
        except InvalidRunTransition as exc:
            raise ServiceError(
                409,
                "INVALID_RUN_TRANSITION",
                str(exc),
            )
        return self._to_response(updated_run)

    def resume_run(
        self,
        run_id: str,
        org_id: str,
        user_id: str,
        request: ResumeRunRequest,
    ) -> RunResponse:
        self._get_authorized_run(run_id, org_id)
        try:
            updated_run = self.repository.resume_run(
                run_id=run_id,
                org_id=org_id,
                actor_id=user_id,
                resume_payload={
                    "resume_token": request.resume_token,
                    "input": request.input,
                },
            )
        except KeyError:
            raise self._not_found()
        except InvalidRunTransition as exc:
            raise ServiceError(409, "INVALID_RUN_TRANSITION", str(exc))
        return self._to_response(updated_run)

    def rerun_run(
        self,
        run_id: str,
        org_id: str,
        user_id: str,
        request: RerunRunRequest,
    ) -> RunResponse:
        source = self._get_authorized_run(run_id, org_id)
        if source["status"] not in {"succeeded", "failed", "cancelled"}:
            raise ServiceError(
                409,
                "RERUN_REQUIRES_TERMINAL_RUN",
                "Only a terminal run can be rerun",
            )

        now = self._now()
        new_run_id = str(uuid4())
        new_input = request.input if request.input is not None else source["input"]
        new_run = {
            "id": new_run_id,
            "parent_run_id": source["id"],
            "org_id": org_id,
            "user_id": user_id,
            "run_type": source["run_type"],
            "idempotency_key": request.idempotency_key,
            "input": new_input,
            "input_hash": self._input_hash(new_input),
            "status": "queued",
            "output": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
            "rerun_reason": request.reason,
        }
        event = self._event(
            new_run_id,
            "run.created",
            {
                "status": "queued",
                "run_type": source["run_type"],
                "parent_run_id": source["id"],
                "rerun_reason": request.reason,
            },
        )
        event["sequence"] = 1
        stored, created = self.repository.create_if_absent(
            org_id,
            new_run,
            event,
        )
        if not created:
            if stored.get("parent_run_id") != source["id"]:
                raise ServiceError(
                    409,
                    "IDEMPOTENCY_CONFLICT",
                    "Idempotency key already belongs to another run",
                )
        return self._to_response(stored)

    def get_trace(self, run_id: str, org_id: str) -> dict[str, Any]:
        trace = self.repository.get_trace(run_id, org_id)
        if trace is None:
            raise self._not_found()
        trace["run"] = self._to_response(trace["run"])
        return trace


run_service = RunService(run_repository)

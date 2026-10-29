"""Refund gateway contract and deterministic local implementation.

The recovery layer depends on two properties of a real payment provider:
the business idempotency key is stable across reruns, and the provider can
look up the original operation by that key.  This fake gateway makes those
properties testable without money movement or a network dependency.
"""

from dataclasses import dataclass
import json
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from typing import Any, Protocol


class RefundGatewayUnavailable(RuntimeError):
    """The provider could not answer a refund status query."""


class HttpRefundGateway:
    """HTTP adapter for a real refund provider.

    Endpoint paths are configurable because every payment provider names its
    lookup endpoint differently.  The recovery contract remains stable:
    write with a business idempotency key, then query by that same key.
    """

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        refund_path: str = "/refunds",
        lookup_path: str = "/refunds/by-idempotency-key/{key}",
        timeout_seconds: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.refund_path = refund_path
        self.lookup_path = lookup_path
        self.timeout_seconds = timeout_seconds

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        headers = {"Accept": "application/json"}
        payload = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            payload = json.dumps(body).encode("utf-8")
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = Request(
            f"{self.base_url}{path}",
            data=payload,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
        except HTTPError as exc:
            if exc.code == 404:
                return None
            raise RefundGatewayUnavailable(
                f"refund provider HTTP {exc.code}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RefundGatewayUnavailable(
                f"refund provider unavailable: {exc}"
            ) from exc
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RefundGatewayUnavailable(
                "refund provider returned invalid JSON"
            ) from exc
        if not isinstance(decoded, dict):
            raise RefundGatewayUnavailable("refund provider response is not an object")
        return decoded

    def refund(
        self,
        *,
        order_id: str,
        ticket_id: str,
        amount: float,
        idempotency_key: str,
    ) -> dict[str, Any]:
        result = self._request(
            "POST",
            self.refund_path,
            {
                "order_id": order_id,
                "ticket_id": ticket_id,
                "amount": amount,
                "idempotency_key": idempotency_key,
            },
        )
        if result is None:
            raise RefundGatewayUnavailable("refund provider returned no result")
        return result

    def get_by_idempotency_key(self, idempotency_key: str) -> dict[str, Any] | None:
        return self._request(
            "GET",
            self.lookup_path.format(key=quote(idempotency_key, safe="")),
        )


class RefundGateway(Protocol):
    def refund(
        self,
        *,
        order_id: str,
        ticket_id: str,
        amount: float,
        idempotency_key: str,
    ) -> dict[str, Any]: ...

    def get_by_idempotency_key(self, idempotency_key: str) -> dict[str, Any] | None: ...


def refund_idempotency_key(order_id: str, ticket_id: str) -> str:
    """Build the business key; deliberately does not include run_id."""
    return f"refund:{order_id}:{ticket_id}:v1"


@dataclass
class FakeRefundGateway:
    """In-memory gateway that behaves like an idempotent payment provider."""

    query_available: bool = True

    def __post_init__(self) -> None:
        self._refunds: dict[str, dict[str, Any]] = {}
        self.request_count = 0

    def refund(
        self,
        *,
        order_id: str,
        ticket_id: str,
        amount: float,
        idempotency_key: str,
    ) -> dict[str, Any]:
        self.request_count += 1
        existing = self._refunds.get(idempotency_key)
        if existing is not None:
            return {**existing, "duplicate": True}

        record = {
            "refund_id": f"RF-{len(self._refunds) + 2201}",
            "status": "succeeded",
            "order_id": order_id,
            "ticket_id": ticket_id,
            "amount": amount,
            "idempotency_key": idempotency_key,
        }
        self._refunds[idempotency_key] = record
        return {**record, "duplicate": False}

    def refund_then_lose_response(
        self,
        *,
        order_id: str,
        ticket_id: str,
        amount: float,
        idempotency_key: str,
    ) -> None:
        """Commit the external effect, then simulate a lost response."""
        self.refund(
            order_id=order_id,
            ticket_id=ticket_id,
            amount=amount,
            idempotency_key=idempotency_key,
        )
        raise TimeoutError("refund response lost after downstream success")

    def get_by_idempotency_key(self, idempotency_key: str) -> dict[str, Any] | None:
        if not self.query_available:
            raise RefundGatewayUnavailable("refund provider query unavailable")
        record = self._refunds.get(idempotency_key)
        return {**record} if record is not None else None

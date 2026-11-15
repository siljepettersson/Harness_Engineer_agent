# Agent Harness

Durable runtime for resumable, observable agent runs.

The service combines a stateful agent loop with durable checkpoints, event
streams, lease-based worker recovery, idempotent tool execution, and
side-effect reconciliation. A small C++ validation layer provides fast,
deterministic checks for event streams, recovery invariants, and refund
idempotency keys.

## Capabilities

- Run lifecycle and transition enforcement
- Durable outbox dispatch through Celery/Redis
- Context checkpoints for crash recovery
- Event replay without model or tool execution
- Lease expiry detection and Sweeper recovery
- Tool-call idempotency and `response_lost` handling
- Periodic reconciliation of uncertain side effects
- Trace and debug APIs for operational investigation

## Requirements

- Python 3.11+
- Redis for Celery transport in deployments that use asynchronous workers
- A downstream refund provider with idempotency-key lookup for live payments
- CMake and a C++17 compiler for native validators

## Local setup

```bash
uv sync --dev
```

Run the API:

```bash
uvicorn main:app --reload
```

Run Python tests:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q
```

## C++ validation layer

```bash
./.venv/bin/cmake -S . -B /tmp/harness-cpp-build -DCMAKE_BUILD_TYPE=Release
./.venv/bin/cmake --build /tmp/harness-cpp-build --parallel
./.venv/bin/ctest --test-dir /tmp/harness-cpp-build --output-on-failure
```

The validator CLI can check a refund key:

```bash
/tmp/harness-cpp-build/cpp/validator_cli \
  key refund:ORD-10086:T-9527:v1 ORD-10086 T-9527
```

It can also validate an event stream from stdin. Each line contains:
`event_id run_id sequence event_type`.

```bash
printf 'e1 run_123 1 run.created\ne2 run_123 2 run.completed\n' \
  | /tmp/harness-cpp-build/cpp/validator_cli events
```

Exit codes are `0` for valid input, `1` for invariant violations, and `2`
for invalid CLI usage.

## Operational endpoints

```text
GET  /runs/{run_id}/trace
GET  /debug/stuck-runs
GET  /debug/replay/{run_id}
GET  /debug/queue
POST /debug/sweep
POST /debug/reconcile
```

Replay is read-only: it reads persisted events and never calls a model,
tool, or downstream business system.

## Configuration and deployment boundaries

`REDIS_URL` configures the Celery/Redis transport and `CELERY_QUEUE` selects
the queue inspected by `/debug/queue`. The refund HTTP adapter accepts a
provider base URL, token, refund path, and idempotency-key lookup path.

The local `FakeRefundGateway` is safe for deterministic tests. It must not be
used for live money movement. Before production deployment, protect all
`/debug/*` routes with service authentication, replace development database
startup with versioned migrations, and configure a real refund provider with
an idempotency-key lookup API.

## CI

The repository workflow runs Python tests, CMake/CTest, and the validator CLI
smoke test on pushes and pull requests.

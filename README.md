# Agent Harness

This project contains a Python AgentRun harness with recovery, replay,
lease-based sweeping, idempotent tool calls, and reconciliation. It also
contains small C++ validators for high-signal consistency checks.

## Python tests

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q
```

## C++ build and tests

```bash
./.venv/bin/cmake -S . -B /tmp/harness-cpp-build -DCMAKE_BUILD_TYPE=Release
./.venv/bin/cmake --build /tmp/harness-cpp-build --parallel
./.venv/bin/ctest --test-dir /tmp/harness-cpp-build --output-on-failure
```

The C++ code checks event ordering, recovery invariants, and refund
idempotency keys. It does not replace the Python state machine or database
transactions.

## Validator CLI

After building, validate a refund key:

```bash
/tmp/harness-cpp-build/cpp/validator_cli \\
  key refund:ORD-10086:T-9527:v1 ORD-10086 T-9527
```

Validate an event stream from stdin. Each line is:
`event_id run_id sequence event_type`.

```bash
printf 'e1 run_123 1 run.created\ne2 run_123 2 run.completed\n' \\
  | /tmp/harness-cpp-build/cpp/validator_cli events
```

The command exits with `0` and prints `VALID` for valid input. Invalid input
prints the violations and exits with `1`; malformed CLI usage exits with `2`.

## Recovery debug endpoints

The Python API exposes:

- `GET /debug/stuck-runs`
- `GET /debug/replay/{run_id}`
- `GET /debug/queue`
- `POST /debug/sweep`
- `POST /debug/reconcile`

Replay is read-only and never calls a model or tool.

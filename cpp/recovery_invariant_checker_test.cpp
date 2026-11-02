#include "recovery_invariant_checker.hpp"

#include <cassert>

using harness::CheckpointSnapshot;
using harness::Event;
using harness::RecoverySnapshot;
using harness::ToolCallSnapshot;
using harness::validate_recovery_snapshot;

int main() {
    const RecoverySnapshot valid{
        "run_123",
        "succeeded",
        {
            {"e1", "run_123", 1, "run.created"},
            {"e2", "run_123", 2, "run.completed"},
        },
        {{1}, {2}},
        {{"tc-refund", "succeeded", "refund:ORD-10086:T-9527:v1", true, true}},
    };
    assert(validate_recovery_snapshot(valid).valid);

    const RecoverySnapshot invalid{
        "run_123",
        "succeeded",
        {{"e1", "run_123", 1, "run.created"}},
        {{2}, {1}},
        {{"tc-refund", "response_lost", "", true, true}},
    };
    const auto result = validate_recovery_snapshot(invalid);
    assert(!result.valid);
    assert(result.errors.size() == 4);

    return 0;
}

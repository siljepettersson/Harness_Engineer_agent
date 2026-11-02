#include "recovery_invariant_checker.hpp"

#include <set>

namespace harness {

namespace {

bool is_terminal_status(const std::string& status) {
    return status == "succeeded" || status == "failed" || status == "cancelled";
}

bool is_terminal_event(const std::string& type) {
    return type == "run.completed" || type == "run.failed" ||
           type == "run.cancelled";
}

}  // namespace

ValidationResult validate_recovery_snapshot(const RecoverySnapshot& snapshot) {
    ValidationResult result = validate_event_stream(snapshot.events);
    auto error = [&result](const std::string& message) {
        result.errors.push_back(message);
    };

    if (!snapshot.events.empty() && snapshot.events.front().run_id != snapshot.run_id) {
        error("first event does not belong to snapshot run");
    }

    if (is_terminal_status(snapshot.run_status)) {
        if (snapshot.events.empty() || !is_terminal_event(snapshot.events.back().type)) {
            error("terminal run is missing a terminal last event");
        }
    }

    int previous_checkpoint = 0;
    for (const CheckpointSnapshot& checkpoint : snapshot.checkpoints) {
        if (checkpoint.index <= previous_checkpoint) {
            error("checkpoint indexes must increase strictly");
        }
        previous_checkpoint = checkpoint.index;
    }

    std::set<std::string> tool_ids;
    for (const ToolCallSnapshot& call : snapshot.tool_calls) {
        if (!tool_ids.insert(call.id).second) {
            error("duplicate tool call id: " + call.id);
        }
        if (call.side_effect && call.idempotency_key.empty()) {
            error("side-effect tool call has no idempotency key: " + call.id);
        }
        if (call.status == "response_lost" && call.has_result) {
            error("response_lost tool call must not have a result: " + call.id);
        }
    }

    result.valid = result.errors.empty();
    return result;
}

}  // namespace harness

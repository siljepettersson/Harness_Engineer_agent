#pragma once

#include "event_sequence_validator.hpp"

#include <string>
#include <vector>

namespace harness {

struct CheckpointSnapshot {
    int index;
};

struct ToolCallSnapshot {
    std::string id;
    std::string status;
    std::string idempotency_key;
    bool side_effect;
    bool has_result;
};

struct RecoverySnapshot {
    std::string run_id;
    std::string run_status;
    std::vector<Event> events;
    std::vector<CheckpointSnapshot> checkpoints;
    std::vector<ToolCallSnapshot> tool_calls;
};

ValidationResult validate_recovery_snapshot(const RecoverySnapshot& snapshot);

}  // namespace harness

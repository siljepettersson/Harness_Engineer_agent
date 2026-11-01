#include "event_sequence_validator.hpp"

#include <set>

namespace harness {

ValidationResult validate_event_stream(const std::vector<Event>& events) {
    ValidationResult result{true, {}};
    if (events.empty()) {
        return result;
    }

    const std::string& expected_run = events.front().run_id;
    int expected_sequence = 1;
    bool terminal = false;
    std::set<std::string> ids;

    for (const Event& event : events) {
        if (event.run_id != expected_run) {
            result.errors.push_back(
                "event " + event.id + " belongs to a different run"
            );
        }
        if (event.sequence != expected_sequence) {
            result.errors.push_back(
                "expected sequence " + std::to_string(expected_sequence) +
                ", got " + std::to_string(event.sequence)
            );
        }
        if (!ids.insert(event.id).second) {
            result.errors.push_back("duplicate event id: " + event.id);
        }
        if (terminal) {
            result.errors.push_back(
                "event after terminal event: " + event.type
            );
        }
        if (event.type == "run.completed" ||
            event.type == "run.failed" ||
            event.type == "run.cancelled") {
            terminal = true;
        }
        expected_sequence = event.sequence + 1;
    }

    result.valid = result.errors.empty();
    return result;
}

}  // namespace harness

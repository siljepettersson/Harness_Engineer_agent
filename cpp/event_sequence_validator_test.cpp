#include "event_sequence_validator.hpp"

#include <cassert>

using harness::Event;
using harness::validate_event_stream;

int main() {
    const std::vector<Event> valid = {
        {"e1", "run_123", 1, "run.created"},
        {"e2", "run_123", 2, "run.started"},
        {"e3", "run_123", 3, "run.completed"},
    };
    assert(validate_event_stream(valid).valid);

    const std::vector<Event> broken_sequence = {
        {"e1", "run_123", 1, "run.created"},
        {"e3", "run_123", 3, "run.completed"},
    };
    assert(!validate_event_stream(broken_sequence).valid);

    const std::vector<Event> duplicate_after_terminal = {
        {"e1", "run_123", 1, "run.created"},
        {"e2", "run_123", 2, "run.completed"},
        {"e2", "run_123", 3, "tool.completed"},
    };
    const auto result = validate_event_stream(duplicate_after_terminal);
    assert(!result.valid);
    assert(result.errors.size() == 2);

    return 0;
}

#pragma once

#include <string>
#include <vector>

namespace harness {

struct Event {
    std::string id;
    std::string run_id;
    int sequence;
    std::string type;
};

struct ValidationResult {
    bool valid;
    std::vector<std::string> errors;
};

ValidationResult validate_event_stream(const std::vector<Event>& events);

}  // namespace harness

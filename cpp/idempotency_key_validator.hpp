#pragma once

#include <string>
#include <vector>

namespace harness {

struct IdempotencyValidationResult {
    bool valid;
    std::vector<std::string> errors;
};

IdempotencyValidationResult validate_refund_key(
    const std::string& key,
    const std::string& order_id,
    const std::string& ticket_id
);

}  // namespace harness

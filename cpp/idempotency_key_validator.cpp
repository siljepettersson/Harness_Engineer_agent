#include "idempotency_key_validator.hpp"

namespace harness {

IdempotencyValidationResult validate_refund_key(
    const std::string& key,
    const std::string& order_id,
    const std::string& ticket_id
) {
    IdempotencyValidationResult result{true, {}};
    const std::string expected =
        "refund:" + order_id + ":" + ticket_id + ":v1";

    if (order_id.empty() || ticket_id.empty()) {
        result.errors.push_back("order_id and ticket_id must be non-empty");
    }
    if (key.empty()) {
        result.errors.push_back("idempotency key must be non-empty");
    } else if (key != expected) {
        result.errors.push_back(
            "refund key must be refund:{order_id}:{ticket_id}:v1"
        );
    }
    if (key.find("run_") != std::string::npos) {
        result.errors.push_back("refund key must not contain run_id");
    }

    result.valid = result.errors.empty();
    return result;
}

}  // namespace harness

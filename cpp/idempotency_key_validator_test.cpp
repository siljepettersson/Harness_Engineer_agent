#include "idempotency_key_validator.hpp"

#include <cassert>

using harness::validate_refund_key;

int main() {
    assert(validate_refund_key(
        "refund:ORD-10086:T-9527:v1", "ORD-10086", "T-9527"
    ).valid);

    const auto run_key = validate_refund_key(
        "refund:run_456:ORD-10086", "ORD-10086", "T-9527"
    );
    assert(!run_key.valid);
    assert(run_key.errors.size() == 2);

    const auto wrong_version = validate_refund_key(
        "refund:ORD-10086:T-9527:v2", "ORD-10086", "T-9527"
    );
    assert(!wrong_version.valid);

    return 0;
}

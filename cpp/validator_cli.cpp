#include "event_sequence_validator.hpp"
#include "idempotency_key_validator.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace {

template <typename Result>
int print_result(const Result& result) {
    if (result.valid) {
        std::cout << "VALID\n";
        return 0;
    }
    std::cout << "INVALID\n";
    for (const std::string& error : result.errors) {
        std::cout << error << "\n";
    }
    return 1;
}

int validate_key(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: validator_cli key <key> <order_id> <ticket_id>\n";
        return 2;
    }
    return print_result(harness::validate_refund_key(argv[2], argv[3], argv[4]));
}

int validate_events() {
    std::vector<harness::Event> events;
    std::string id;
    std::string run_id;
    std::string type;
    int sequence = 0;
    while (std::cin >> id >> run_id >> sequence >> type) {
        events.push_back({id, run_id, sequence, type});
    }
    return print_result(harness::validate_event_stream(events));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: validator_cli key|events ...\n";
        return 2;
    }
    const std::string command = argv[1];
    if (command == "key") {
        return validate_key(argc, argv);
    }
    if (command == "events") {
        return validate_events();
    }
    std::cerr << "unknown command: " << command << "\n";
    return 2;
}

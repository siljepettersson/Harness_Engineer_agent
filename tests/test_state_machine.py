from state_machine import ALLOWED_RUN_TRANSITIONS, RUN_TERMINAL_STATUSES


def test_run_transition_whitelist_contains_only_the_nine_supported_edges():
    assert ALLOWED_RUN_TRANSITIONS == {
        "queued": {"running", "cancelled"},
        "running": {
            "queued",
            "waiting_for_user",
            "succeeded",
            "failed",
            "cancelled",
        },
        "waiting_for_user": {"queued", "cancelled"},
        "succeeded": set(),
        "failed": set(),
        "cancelled": set(),
    }


def test_terminal_statuses_have_no_outgoing_edges():
    assert RUN_TERMINAL_STATUSES == {"succeeded", "failed", "cancelled"}
    for status in RUN_TERMINAL_STATUSES:
        assert ALLOWED_RUN_TRANSITIONS[status] == set()

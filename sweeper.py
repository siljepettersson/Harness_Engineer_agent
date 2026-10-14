from time import sleep

from repositories.run_repository import run_repository


def sweep_once(limit: int = 100) -> int:
    """Recover runs whose worker lease expired before completion."""
    return run_repository.recover_expired_leases(limit=limit)


def run_forever(poll_interval_seconds: float = 10.0) -> None:
    """Continuously recover expired leases for dispatcher pickup."""
    while True:
        sweep_once()
        sleep(poll_interval_seconds)


if __name__ == "__main__":
    run_forever()

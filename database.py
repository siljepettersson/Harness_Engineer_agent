from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


DATABASE_URL = "sqlite:///./agent_harness.db"


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""


engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    """Provide one database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create tables for the current development database.

    The model module must be imported before create_all so SQLAlchemy knows
    about every mapped table. Alembic will replace this mechanism later.
    """
    from models import (  # noqa: F401
        AgentRun,
        AgentStep,
        AuditLog,
        Checkpoint,
        ModelCall,
        OutboxJob,
        RunEvent,
        ToolCall,
        UsageRecord,
    )

    Base.metadata.create_all(bind=engine)

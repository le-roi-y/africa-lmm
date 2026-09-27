from __future__ import annotations

from collections.abc import Generator

from sqlalchemy.orm import Session

from .connection import engine


def get_db() -> Generator[Session, None, None]:
    """Provide a SQLAlchemy session for a FastAPI request."""
    with Session(engine) as session:
        yield session

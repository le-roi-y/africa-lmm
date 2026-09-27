from __future__ import annotations

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase

load_dotenv()


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://africa_lmm:africa_lmm_dev_2026@localhost:5432/africa_lmm",
)


class Base(DeclarativeBase):
    """Base SQLAlchemy pour les modèles AFRICA-LMM."""


engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

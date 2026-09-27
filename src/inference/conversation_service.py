from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from src.database.models import Conversation, Message
from src.inference.title_generator import generate_conversation_title


class ConversationService:
    """Service de persistence des conversations AFRICA-LMM."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create(
        self,
        title: str = "Nouvelle conversation",
    ) -> Conversation:
        conversation = Conversation(
            title=title.strip() or "Nouvelle conversation",
        )

        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)

        return conversation

    def list(self) -> list[Conversation]:
        statement = (
            select(Conversation)
            .order_by(Conversation.updated_at.desc())
        )

        return list(self.db.scalars(statement).all())

    def get(
        self,
        conversation_id: str,
    ) -> Conversation | None:
        try:
            conversation_uuid = uuid.UUID(conversation_id)
        except ValueError:
            return None

        statement = (
            select(Conversation)
            .where(Conversation.id == conversation_uuid)
            .options(selectinload(Conversation.messages))
        )

        return self.db.scalars(statement).first()

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        citations: list[dict[str, Any]] | None = None,
    ) -> Message:
        conversation = self.get(conversation_id)

        if conversation is None:
            raise ValueError("Conversation not found.")

        if role == "user" and conversation.title == "Nouvelle conversation":
            conversation.title = generate_conversation_title(content)

        message = Message(
            conversation_id=conversation.id,
            role=role,
            content=content,
            citations=citations or [],
        )

        self.db.add(message)
        self.db.commit()
        self.db.refresh(message)

        return message

    def recent_messages(
        self,
        conversation_id: str,
        limit: int = 10,
    ) -> list[Message]:
        """Retourne les derniers messages d'une conversation dans l'ordre chronologique."""
        conversation = self.get(conversation_id)

        if conversation is None:
            return []

        messages = sorted(
            conversation.messages,
            key=lambda message: message.created_at,
        )

        return messages[-limit:]

    def delete(
        self,
        conversation_id: str,
    ) -> bool:
        conversation = self.get(conversation_id)

        if conversation is None:
            return False

        self.db.delete(conversation)
        self.db.commit()

        return True

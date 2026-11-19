
from datetime import datetime, timezone
import uuid

from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    Text,
    DateTime,
    ForeignKey,
    JSON
)

from sqlalchemy.orm import relationship

from database import Base


def utc_now():
    return datetime.now(timezone.utc)


# ==========================================
# USER TABLE
# ==========================================

class User(Base):

    __tablename__ = "users"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    name = Column(
        String(100),
        nullable=False
    )

    email = Column(
        String(255),
        unique=True,
        nullable=False,
        index=True
    )

    password_hash = Column(
        String(255),
        nullable=False
    )

    created_at = Column(
        DateTime(timezone=True),
        default=utc_now
    )

    chats = relationship(
        "Chat",
        back_populates="user",
        cascade="all, delete-orphan"
    )


# ==========================================
# CHAT TABLE
# ==========================================

class Chat(Base):

    __tablename__ = "chats"

    id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )

    user_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False,
        index=True
    )

    title = Column(
        String(255),
        default="New chat",
        nullable=False
    )

    is_pinned = Column(
        Boolean,
        default=False,
        nullable=False
    )

    created_at = Column(
        DateTime(timezone=True),
        default=utc_now
    )

    updated_at = Column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now
    )

    user = relationship(
        "User",
        back_populates="chats"
    )

    messages = relationship(
        "Message",
        back_populates="chat",
        cascade="all, delete-orphan",
        order_by="Message.created_at"
    )


# ==========================================
# MESSAGE TABLE
# ==========================================

class Message(Base):

    __tablename__ = "messages"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    chat_id = Column(
        String(36),
        ForeignKey("chats.id"),
        nullable=False,
        index=True
    )

    role = Column(
        String(20),
        nullable=False
    )

    content = Column(
        Text,
        nullable=False
    )

    sources = Column(
        JSON,
        default=list
    )

    created_at = Column(
        DateTime(timezone=True),
        default=utc_now
    )

    chat = relationship(
        "Chat",
        back_populates="messages"
    )


# ==========================================
# DATABASE INITIALIZATION
# ==========================================

if __name__ == "__main__":

    from database import engine

    Base.metadata.create_all(bind=engine)

    print("Database tables created successfully!")

"""
app.py - FastAPI Backend for darglobal-bot
Provides REST API for Authentication, Persistent Chats, Messages, and Hybrid RAG.
"""

import os
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Request, Response, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from database import get_db, Base, engine
from models import User, Chat, Message
from auth import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token
)
from rag import get_rag
from external_search import search_external
from llm import generate_answer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("darglobal_bot.app")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
load_dotenv(BASE_DIR / ".env")

# Initialize database tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI App
app = FastAPI(
    title="darglobal-bot API",
    description="Independent AI Real Estate Informational Assistant for DarGlobal & Wasalt",
    version="2.0.0"
)

# Initialize RAG index at startup
rag = get_rag()


# ==========================================
# AUTHENTICATION DEPENDENCY
# ==========================================

def get_current_user(
    request: Request,
    db: Session = Depends(get_db)
) -> User:
    """Retrieve authenticated user from HTTP-only cookie or Authorization header."""
    token = request.cookies.get("access_token")

    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1]

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    user_id = decode_access_token(token)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session token"
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    return user


def get_optional_user(
    request: Request,
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Retrieve user if authenticated, else None (for hybrid public/private endpoints)."""
    token = request.cookies.get("access_token")
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1]

    if not token:
        return None

    user_id = decode_access_token(token)
    if user_id is None:
        return None

    return db.query(User).filter(User.id == user_id).first()


def get_user_chat(chat_id: str, user_id: int, db: Session) -> Chat:
    """Enforce strict chat ownership validation."""
    chat = db.query(Chat).filter(
        Chat.id == chat_id,
        Chat.user_id == user_id
    ).first()

    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found or access denied"
        )
    return chat


# ==========================================
# PYDANTIC SCHEMAS
# ==========================================

class RegisterRequest(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class CreateChatRequest(BaseModel):
    title: Optional[str] = Field(default="New chat", max_length=255)


class UpdateChatRequest(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=255)
    is_pinned: Optional[bool] = None


class SendMessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: List[Dict[str, Any]] = Field(default_factory=list)
    chat_id: Optional[str] = None


# ==========================================
# HEALTH & METRICS ENDPOINT
# ==========================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "app_name": "darglobal-bot",
        "version": "2.0.0",
        "total_chunks": len(rag.chunks) if rag else 0,
        "bm25_ready": rag.bm25 is not None if rag else False,
        "semantic_ready": rag.chunk_embeddings is not None if rag else False
    }


# ==========================================
# AUTHENTICATION API
# ==========================================

@app.post("/api/register")
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    name = req.name.strip()
    email = req.email.lower().strip()
    password = req.password

    if len(name) < 2:
        raise HTTPException(status_code=400, detail="Name cannot be empty")

    existing_user = db.query(User).filter(User.email == email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email address already exists"
        )

    user = User(
        name=name,
        email=email,
        password_hash=hash_password(password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    logger.info(f"New user registered: {email}")
    return {
        "message": "Registration successful",
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email
        }
    }


@app.post("/api/login")
def login(req: LoginRequest, response: Response, db: Session = Depends(get_db)):
    email = req.email.lower().strip()
    user = db.query(User).filter(User.email == email).first()

    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    token = create_access_token(user.id)

    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        secure=bool(os.getenv("COOKIE_SECURE", "").lower() in ("true", "1", "yes")),
        samesite="lax",
        max_age=60 * 60 * 24 * 7,  # 7 days
        path="/"
    )

    logger.info(f"User logged in: {email}")
    return {
        "message": "Login successful",
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email
        }
    }


@app.post("/api/logout")
def logout(response: Response):
    response.delete_cookie(key="access_token", path="/")
    return {"message": "Logged out successfully"}


@app.get("/api/me")
def get_me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "name": current_user.name,
        "email": current_user.email
    }


# ==========================================
# CHAT MANAGEMENT API
# ==========================================

@app.get("/api/chats")
def list_chats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    chats = db.query(Chat).filter(
        Chat.user_id == current_user.id
    ).order_by(
        Chat.is_pinned.desc(),
        Chat.updated_at.desc()
    ).all()

    return [
        {
            "id": chat.id,
            "title": chat.title,
            "is_pinned": chat.is_pinned,
            "created_at": chat.created_at.isoformat(),
            "updated_at": chat.updated_at.isoformat()
        }
        for chat in chats
    ]


@app.post("/api/chats")
def create_chat(
    req: CreateChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    title = (req.title or "New chat").strip()
    chat = Chat(
        user_id=current_user.id,
        title=title
    )
    db.add(chat)
    db.commit()
    db.refresh(chat)

    return {
        "id": chat.id,
        "title": chat.title,
        "is_pinned": chat.is_pinned,
        "created_at": chat.created_at.isoformat(),
        "updated_at": chat.updated_at.isoformat()
    }


@app.get("/api/chats/{chat_id}")
def get_chat_details(
    chat_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    chat = get_user_chat(chat_id, current_user.id, db)

    messages_data = []
    for msg in chat.messages:
        # Handle sources whether saved as list or rich dict
        sources_list = []
        answer_type = "internal"
        external_sources = []
        retrieval_info = {}

        if isinstance(msg.sources, dict):
            sources_list = msg.sources.get("sources", [])
            answer_type = msg.sources.get("answer_type", "internal")
            external_sources = msg.sources.get("external_sources", [])
            retrieval_info = msg.sources.get("retrieval_info", {})
        elif isinstance(msg.sources, list):
            sources_list = msg.sources

        messages_data.append({
            "id": msg.id,
            "role": msg.role,
            "content": msg.content,
            "sources": sources_list,
            "answer_type": answer_type,
            "external_sources": external_sources,
            "retrieval_info": retrieval_info,
            "created_at": msg.created_at.isoformat()
        })

    return {
        "id": chat.id,
        "title": chat.title,
        "is_pinned": chat.is_pinned,
        "created_at": chat.created_at.isoformat(),
        "updated_at": chat.updated_at.isoformat(),
        "messages": messages_data
    }


@app.patch("/api/chats/{chat_id}")
def update_chat(
    chat_id: str,
    req: UpdateChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    chat = get_user_chat(chat_id, current_user.id, db)

    if req.title is not None:
        clean_title = req.title.strip()
        if clean_title:
            chat.title = clean_title

    if req.is_pinned is not None:
        chat.is_pinned = req.is_pinned

    db.commit()
    db.refresh(chat)

    return {
        "id": chat.id,
        "title": chat.title,
        "is_pinned": chat.is_pinned,
        "created_at": chat.created_at.isoformat(),
        "updated_at": chat.updated_at.isoformat()
    }


@app.post("/api/chats/{chat_id}/pin")
def toggle_pin_chat(
    chat_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    chat = get_user_chat(chat_id, current_user.id, db)
    chat.is_pinned = not chat.is_pinned
    db.commit()
    db.refresh(chat)

    return {
        "id": chat.id,
        "title": chat.title,
        "is_pinned": chat.is_pinned
    }


@app.delete("/api/chats/{chat_id}")
def delete_chat(
    chat_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    chat = get_user_chat(chat_id, current_user.id, db)
    db.delete(chat)
    db.commit()
    return {"message": "Chat deleted successfully", "id": chat_id}


# ==========================================
# CORE RAG ORCHESTRATION PIPELINE
# ==========================================

def run_rag_pipeline(query: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Executes the complete Hybrid RAG + External Search + LLM generation pipeline:
    1. Hybrid BM25 + Semantic vector retrieval with RRF ranking.
    2. Evaluates context sufficiency.
    3. If insufficient, triggers external search fallback.
    4. Passes enriched context to LLM.
    5. Returns standardized response schema with source transparency.
    """
    clean_query = query.strip()
    retrieval_query = clean_query
    if history and (len(clean_query.split()) <= 7 or re.search(r"\b(it|its|this|that|those|them|they|there|same)\b", clean_query, re.I)):
        recent_context = " ".join(
            item.get("content", "").strip()[:700]
            for item in history[-4:]
            if item.get("role") in ("user", "assistant") and isinstance(item.get("content"), str)
        )
        if recent_context:
            retrieval_query = f"{recent_context} {clean_query}"[:2400]

    # Step 1: Hybrid retrieval from internal knowledge base
    try:
        retrieval_output = rag.retrieve(retrieval_query, top_k=6)
        internal_chunks = retrieval_output["results"]
        is_sufficient = retrieval_output["sufficient"]

        external_results = []
        if not is_sufficient:
            logger.info("Internal knowledge insufficient; checking configured external search.")
            external_results = search_external(retrieval_query, max_results=3)

        ai_response = generate_answer(
            query=clean_query,
            internal_chunks=internal_chunks,
            external_results=external_results,
            conversation_history=history,
            sufficient_internal=is_sufficient
        )
    except Exception as exc:
        logger.exception("Answer pipeline failed")
        raise HTTPException(status_code=503, detail="The answer service is temporarily unavailable. Please try again.") from exc

    if not isinstance(ai_response, dict) or not isinstance(ai_response.get("answer"), str) or not ai_response["answer"].strip():
        raise HTTPException(status_code=502, detail="The answer service returned an empty response. Please try again.")
    return ai_response


# ==========================================
# CHAT MESSAGING ENDPOINTS
# ==========================================

@app.post("/api/chats/{chat_id}/messages")
def post_chat_message(
    chat_id: str,
    req: SendMessageRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Send message within an existing persistent chat.
    Saves user message, generates answer via Hybrid RAG, saves assistant message.
    """
    chat = get_user_chat(chat_id, current_user.id, db)
    user_text = req.message.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Use persisted history as the source of truth for follow-up turns.
    history = [
        {"role": m.role, "content": m.content}
        for m in chat.messages[-10:]
    ]
    ai_response = run_rag_pipeline(user_text, history)

    db.add(Message(chat_id=chat.id, role="user", content=user_text, sources=[]))
    if chat.title == "New chat" or not chat.messages:
        chat.title = user_text[:35] + ("..." if len(user_text) > 35 else "")

    # Persist assistant message with metadata
    assistant_msg = Message(
        chat_id=chat.id,
        role="assistant",
        content=ai_response["answer"],
        sources={
            "sources": ai_response["sources"],
            "answer_type": ai_response["answer_type"],
            "external_sources": ai_response["external_sources"],
            "retrieval_info": ai_response["retrieval_info"]
        }
    )
    db.add(assistant_msg)
    db.commit()
    db.refresh(chat)

    return {
        "answer": ai_response["answer"],
        "sources": ai_response["sources"],
        "answer_type": ai_response["answer_type"],
        "external_sources": ai_response["external_sources"],
        "retrieval_info": ai_response["retrieval_info"],
        "chat_id": chat.id,
        "chat_title": chat.title
    }


@app.post("/chat")
def standalone_chat(
    req: ChatRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Standard /chat endpoint adhering strictly to response schema.
    Supports anonymous exploratory queries as well as authenticated sessions.
    """
    user_text = req.message.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    user = get_optional_user(request, db)

    # If chat_id is provided and user is authenticated, persist the turn
    active_chat = None
    if req.chat_id and user:
        active_chat = db.query(Chat).filter(
            Chat.id == req.chat_id,
            Chat.user_id == user.id
        ).first()
        if not active_chat:
            raise HTTPException(status_code=404, detail="Conversation not found or access denied")

    history = req.history[-10:]
    if active_chat:
        history = [{"role": m.role, "content": m.content} for m in active_chat.messages[-10:]]
    ai_response = run_rag_pipeline(user_text, history)

    if active_chat:
        db.add(Message(chat_id=active_chat.id, role="user", content=user_text, sources=[]))
        if active_chat.title == "New chat":
            active_chat.title = user_text[:35] + ("..." if len(user_text) > 35 else "")
        assistant_msg = Message(
            chat_id=active_chat.id,
            role="assistant",
            content=ai_response["answer"],
            sources={
                "sources": ai_response["sources"],
                "answer_type": ai_response["answer_type"],
                "external_sources": ai_response["external_sources"],
                "retrieval_info": ai_response["retrieval_info"]
            }
        )
        db.add(assistant_msg)
        db.commit()

    return {
        "answer": ai_response["answer"],
        "sources": ai_response["sources"],
        "answer_type": ai_response["answer_type"],
        "external_sources": ai_response["external_sources"],
        "retrieval_info": ai_response["retrieval_info"]
    }


# ==========================================
# STATIC FRONTEND SERVING
# ==========================================

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(STATIC_DIR / "index.html")

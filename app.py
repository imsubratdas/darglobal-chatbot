
import os
import json
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE = Path(__file__).resolve().parent

load_dotenv(BASE / ".env")

API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

# Use only free routing options.
DEFAULT_MODELS = "openrouter/free"

MODELS = [
    model.strip()
    for model in os.getenv(
        "OPENROUTER_MODELS",
        DEFAULT_MODELS
    ).split(",")
    if model.strip()
]

OPENROUTER_URL = (
    "https://openrouter.ai/api/v1/chat/completions"
)

DATA_FILE = BASE / "data" / "pages.json"
STATIC_DIR = BASE / "static"


# ==========================================
# 2. LOAD DATA
# ==========================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}"
    )

with open(DATA_FILE, "r", encoding="utf-8") as file:
    pages = json.load(file)

if not isinstance(pages, list):
    raise ValueError("pages.json must contain a JSON list")


# ==========================================
# 3. TEXT CHUNKING
# ==========================================

def split_text(text, size=900, overlap=150):

    output = []

    start = 0

    while start < len(text):

        piece = text[start:start + size].strip()

        if piece:
            output.append(piece)

        start += size - overlap

    return output


# ==========================================
# 4. CREATE KNOWLEDGE BASE
# ==========================================

chunks = []

for page in pages:

    text = page.get("text", "")

    if not isinstance(text, str) or not text.strip():
        continue

    for piece in split_text(text):

        chunks.append({
            "company": page.get("company", "Unknown"),
            "url": page.get("url", ""),
            "title": page.get("title", ""),
            "text": piece
        })


# ==========================================
# 5. TF-IDF VECTOR SEARCH
# ==========================================

vectorizer = None
matrix = None

if chunks:

    documents = [
        (
            chunk["company"] + " " +
            chunk["title"] + " " +
            chunk["text"]
        )
        for chunk in chunks
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=50000
    )

    matrix = vectorizer.fit_transform(documents)

    print(f"Loaded {len(pages)} pages")
    print(f"Created {len(chunks)} text chunks")

else:

    print("WARNING: No searchable documents found")


# ==========================================
# 6. SYSTEM PROMPT
# ==========================================

SYSTEM_PROMPT = """
You are PropIntel AI, a real estate intelligence
assistant for DarGlobal and Wasalt.

Your responsibilities:

1. Answer questions using the supplied context.
2. Identify whether information belongs to DarGlobal
   or Wasalt.
3. Never invent prices, locations, amenities,
   property details or project information.
4. If the answer is unavailable in the context,
   explicitly say that you do not have that information.
5. Distinguish between property listings and
   development projects.
6. Do not claim that listing availability is live.
7. Provide clear and concise answers.
8. Use the supplied source references to support
   factual statements.

Only use the retrieved context as factual evidence.
"""


# ==========================================
# 7. RETRIEVAL FUNCTION
# ==========================================

def retrieve(query, k=5):

    if vectorizer is None or matrix is None:
        return []

    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        matrix
    )[0]

    top_indices = scores.argsort()[::-1][:k]

    results = []

    for index in top_indices:

        if scores[index] <= 0:
            continue

        result = chunks[index].copy()

        result["score"] = float(scores[index])

        results.append(result)

    return results


# ==========================================
# 8. FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="PropIntel AI",
    description="DarGlobal and Wasalt AI Assistant",
    version="1.0.0"
)


# ==========================================
# 9. REQUEST SCHEMA
# ==========================================

class ChatRequest(BaseModel):

    message: str

    history: list[dict] = Field(
        default_factory=list
    )


# ==========================================
# 10. HEALTH ENDPOINT
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "pages": len(pages),
        "chunks": len(chunks),
        "models": MODELS
    }


# ==========================================
# 11. CHAT ENDPOINT
# ==========================================

@app.post("/chat")
def chat(req: ChatRequest):

    # Check API key
    if not API_KEY:

        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY is not configured"
        )

    question = req.message.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty"
        )

    if len(question) > 2000:

        raise HTTPException(
            status_code=400,
            detail="Message is too long"
        )

    # Retrieve relevant context
    hits = retrieve(question, k=5)

    context_parts = []

    for hit in hits:

        context_parts.append(
            f"""
Source: {hit['company']}
Title: {hit['title']}
URL: {hit['url']}

Content:
{hit['text']}
"""
        )

    context = "\n\n".join(context_parts)

    if not context:

        context = (
            "No relevant information was found "
            "in the collected dataset."
        )

    # Prepare messages
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT + "\n\nCONTEXT:\n" + context
        }
    ]

    # Limit conversation history
    for item in req.history[-6:]:

        role = item.get("role")
        content = item.get("content")

        if (
            role in ("user", "assistant")
            and isinstance(content, str)
            and content.strip()
        ):

            messages.append({
                "role": role,
                "content": content[:3000]
            })

    messages.append({
        "role": "user",
        "content": question
    })

    # ======================================
    # 12. OPENROUTER REQUEST
    # ======================================

    last_error = ""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "PropIntel AI"
    }

    for model in MODELS:

        try:

            response = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.2,
                    "max_tokens": 1000
                },
                timeout=60
            )

            if response.status_code == 200:

                data = response.json()

                answer = (
                    data["choices"][0]["message"]["content"]
                )

                sources = []

                for hit in hits:

                    if hit["url"] and hit["url"] not in sources:
                        sources.append(hit["url"])

                return {
                    "answer": answer,
                    "sources": sources,
                    "model": data.get("model", model),
                    "retrieved_chunks": len(hits)
                }

            last_error = (
                f"{model}: HTTP {response.status_code} - "
                f"{response.text[:500]}"
            )

            print("[OPENROUTER ERROR]", last_error)

        except requests.RequestException as error:

            last_error = f"{model}: {str(error)}"

            print("[REQUEST ERROR]", last_error)

        except (ValueError, KeyError, IndexError) as error:

            last_error = f"{model}: Invalid API response: {error}"

            print("[RESPONSE ERROR]", last_error)

    raise HTTPException(
        status_code=502,
        detail="All configured free models failed: " + last_error
    )


# ==========================================
# 13. STATIC FRONTEND
# ==========================================

if not STATIC_DIR.exists():

    raise FileNotFoundError(
        f"Static folder not found: {STATIC_DIR}"
    )

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)


@app.get("/")
def home():

    return FileResponse(
        STATIC_DIR / "index.html"
    )

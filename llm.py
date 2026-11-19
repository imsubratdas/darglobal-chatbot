"""
llm.py - LLM Generation & Orchestration for darglobal-bot
Handles OpenRouter API requests, multi-model fallback, content extraction,
and grounds responses with internal and external evidence.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
logger = logging.getLogger("darglobal_bot.llm")

API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Robust list of reliable free models with fallback
DEFAULT_MODELS = "openrouter/free"

configured_models = os.getenv("OPENROUTER_MODELS", DEFAULT_MODELS)
MODELS = [m.strip() for m in configured_models.split(",") if m.strip()]

SYSTEM_PROMPT = """You are darglobal-bot, an independent AI real estate informational assistant designed to help users explore DarGlobal luxury developments and Wasalt real estate marketplace listings.

CORE OPERATIONAL RULES:
1. Brand & Identity:
   - Your name is darglobal-bot.
   - You are an independent AI informational assistant, not an official spokesperson or direct product of DarGlobal or Wasalt.
   - Maintain a sophisticated, helpful, professional luxury real-estate advisory tone.

2. Factual Grounding:
   - Use the provided context (internal knowledge base and/or external research) to answer factual questions accurately.
   - Clearly identify whether the discussed property or project belongs to DarGlobal (luxury developer portfolio) or Wasalt (Saudi real estate portal and marketplace).
   - Never fabricate prices, square footage, amenities, launch dates, or locations not present in the context.
   - If specific details (e.g., exact live listing availability or pricing) are absent from the context, acknowledge this transparently.

3. Context & Follow-up Understanding:
   - Maintain context from previous conversation turns.
   - Properly resolve pronouns and references such as "it", "that development", "where is it located?", or "what are its amenities?".
   - Avoid unnecessary robotic repetition; answer directly and concisely with elegant markdown formatting (bullet points, bold highlights, clean sections).

4. Source Transparency:
   - Reference the provided sources when citing projects, locations, or market figures.
"""


def extract_message_content(data: Dict[str, Any]) -> Optional[str]:
    """
    Safely extract text content from OpenRouter response payload.
    Handles regular content, reasoning tokens, text fields, and empty answers.
    """
    choices = data.get("choices")
    if not choices or not isinstance(choices, list) or len(choices) == 0:
        return None

    message = choices[0].get("message", {})
    if not isinstance(message, dict):
        return None

    # Primary content field
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts = [item.get("text", "") for item in content if isinstance(item, dict) and isinstance(item.get("text"), str)]
        combined = "\n".join(part.strip() for part in parts if part.strip())
        if combined:
            return combined

    # Do not expose hidden reasoning traces as user-facing answer content.
    text = message.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    return None


def generate_answer(
    query: str,
    internal_chunks: List[Dict[str, Any]],
    external_results: List[Dict[str, Any]],
    conversation_history: List[Dict[str, Any]],
    sufficient_internal: bool = True
) -> Dict[str, Any]:
    """
    Generate a grounded response using internal RAG chunks and external fallback if applicable.
    Standardizes output according to the application contract:
    {
      "answer": "...",
      "sources": [...],
      "answer_type": "internal" | "external" | "mixed",
      "external_sources": [...],
      "retrieval_info": { ... }
    }
    """
    # Build context string
    context_sections = []

    # Internal context
    internal_sources = []
    if internal_chunks:
        internal_text_parts = []
        for i, chunk in enumerate(internal_chunks, 1):
            url = chunk.get("url", "")
            if url and not any(source["url"] == url for source in internal_sources):
                internal_sources.append({
                    "url": url, "title": chunk.get("title") or url,
                    "company": chunk.get("company"), "chunk_id": chunk.get("id"),
                    "source_type": "internal"
                })

            internal_text_parts.append(
                f"[Internal Source {i}] Company: {chunk.get('company')} | Title: {chunk.get('title')}\n"
                f"URL: {url}\n"
                f"Content: {chunk.get('text')}\n"
            )
        context_sections.append("=== INTERNAL KNOWLEDGE BASE CONTEXT ===\n" + "\n".join(internal_text_parts))

    # External context
    external_sources = []
    if external_results:
        external_text_parts = []
        for j, ext in enumerate(external_results, 1):
            url = ext.get("url", "")
            if url and not any(source["url"] == url for source in external_sources):
                external_sources.append({
                    "url": url, "title": ext.get("title") or url,
                    "source_type": "external"
                })

            external_text_parts.append(
                f"[External Research {j}] Title: {ext.get('title')}\n"
                f"URL: {url}\n"
                f"Snippet: {ext.get('snippet')}\n"
            )
        context_sections.append("=== EXTERNAL WEB RESEARCH CONTEXT ===\n" + "\n".join(external_text_parts))

    combined_context = "\n\n".join(context_sections).strip()[:12000]
    if not combined_context:
        combined_context = "No specific reference documents found in the database or external search."

    # Determine answer type
    if internal_chunks and external_results:
        answer_type = "mixed"
    elif external_results:
        answer_type = "external"
    elif internal_chunks:
        answer_type = "internal"
    else:
        answer_type = "general"

    # Assemble messages for LLM
    system_message = SYSTEM_PROMPT + "\n\nCONTEXT INFORMATION:\n" + combined_context
    messages = [{"role": "system", "content": system_message}]

    # Include recent conversation turns (last 6 messages)
    for msg in conversation_history[-6:]:
        role = msg.get("role")
        content = msg.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content[:2500]})

    # Add current user query
    messages.append({"role": "user", "content": query})

    # Call OpenRouter with model fallback loop
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://darglobal-bot.local",
        "X-Title": "darglobal-bot"
    }

    last_error = ""
    model_used = None
    final_answer = None

    if API_KEY:
        for model in MODELS:
            try:
                logger.info(f"Attempting completion with model: {model}")
                resp = requests.post(
                    OPENROUTER_URL,
                    headers=headers,
                    json={
                        "model": model,
                        "messages": messages,
                        "temperature": 0.25,
                        "max_tokens": 1200
                    },
                    timeout=35
                )

                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except ValueError:
                        data = {}
                    content = extract_message_content(data)
                    if content:
                        final_answer = content
                        model_used = data.get("model", model)
                        break
                    else:
                        last_error = f"{model}: Empty or null content in response: {resp.text[:300]}"
                        logger.warning(last_error)
                else:
                    last_error = f"{model}: HTTP {resp.status_code} - {resp.text[:300]}"
                    logger.warning(last_error)
            except Exception as e:
                last_error = f"{model}: Exception - {str(e)}"
                logger.warning(last_error)

    # If all models failed or API key is not configured, generate a deterministic grounded response
    if not final_answer:
        logger.error(f"All LLM models failed or API key missing. Last error: {last_error}")
        if internal_chunks:
            # Provide grounded synthesis from retrieved chunks
            top_hit = internal_chunks[0]
            final_answer = (
                f"**Information retrieved from the {top_hit.get('company')} knowledge base:**\n\n"
                f"{top_hit.get('text')}\n\n"
                f"*Reference: [{top_hit.get('title')}]({top_hit.get('url')})*"
            )
            model_used = "knowledge-base-fallback"
        elif external_results:
            top_hit = external_results[0]
            final_answer = (
                "The language model is temporarily unavailable. External search returned this relevant result: "
                f"**{top_hit.get('title') or 'Web result'}** — {top_hit.get('snippet') or 'No summary was provided.'}"
            )
            model_used = "external-search-fallback"
        else:
            final_answer = (
                "I was unable to retrieve specific information for this query from the knowledge base, "
                "and the language model service is currently unavailable. Please verify your query or try again shortly."
            )
            model_used = "system-fallback"

    # All sources combined
    all_sources = internal_sources + external_sources

    return {
        "answer": final_answer,
        "sources": all_sources,
        "answer_type": answer_type,
        "external_sources": external_sources,
        "retrieval_info": {
            "internal_context_used": bool(internal_chunks),
            "external_search_used": bool(external_results),
            "internal_chunks_count": len(internal_chunks),
            "external_results_count": len(external_results),
            "model_used": model_used or "unknown",
            "internal_sufficient": bool(sufficient_internal)
        }
    }

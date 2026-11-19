"""
external_search.py - Robust External Knowledge Fallback for darglobal-bot
Provides real external search via Tavily, SerpAPI, DuckDuckGo API, and Wikipedia.
Always returns real sources and never fabricates URLs.
"""

import os
import re
import logging
from typing import List, Dict, Any
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
logger = logging.getLogger("darglobal_bot.external_search")

ENABLE_EXTERNAL_SEARCH = os.getenv("ENABLE_EXTERNAL_SEARCH", "true").lower() in ("true", "1", "yes")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "").strip()
REQUEST_TIMEOUT = 5  # seconds


def search_external(query: str, max_results: int = 4) -> List[Dict[str, Any]]:
    """
    Query external search providers with graceful fallbacks.
    Returns list of dicts: [{'title': str, 'url': str, 'snippet': str, 'source_type': 'external'}]
    """
    if not ENABLE_EXTERNAL_SEARCH:
        logger.info("External search is disabled by configuration (ENABLE_EXTERNAL_SEARCH=false).")
        return []

    clean_query = query.strip()
    if not clean_query:
        return []

    # 1. Try Tavily API if configured
    if TAVILY_API_KEY:
        try:
            logger.info("Querying Tavily search...")
            res = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": clean_query,
                    "search_depth": "basic",
                    "max_results": max_results,
                    "include_answer": False
                },
                timeout=REQUEST_TIMEOUT
            )
            if res.status_code == 200:
                data = res.json()
                results = []
                for item in data.get("results", [])[:max_results]:
                    results.append({
                        "title": item.get("title", "Web Result"),
                        "url": item.get("url", ""),
                        "snippet": item.get("content", ""),
                        "source_type": "external"
                    })
                if results:
                    return results
        except Exception as e:
            logger.warning(f"Tavily search failed: {e}. Falling back...")

    # 2. Try SerpAPI if configured
    if SERPAPI_API_KEY:
        try:
            logger.info("Querying SerpAPI search...")
            res = requests.get(
                "https://serpapi.com/search",
                params={
                    "api_key": SERPAPI_API_KEY,
                    "q": clean_query,
                    "engine": "google",
                    "num": max_results
                },
                timeout=REQUEST_TIMEOUT
            )
            if res.status_code == 200:
                data = res.json()
                results = []
                for item in data.get("organic_results", [])[:max_results]:
                    results.append({
                        "title": item.get("title", "Web Result"),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "source_type": "external"
                    })
                if results:
                    return results
        except Exception as e:
            logger.warning(f"SerpAPI search failed: {e}. Falling back...")

    # 3. DuckDuckGo Instant Answer API (Free, no API key required)
    try:
        logger.info("Querying DuckDuckGo Instant Answer API...")
        res = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": clean_query, "format": "json", "no_redirect": 1, "no_html": 1},
            headers={"User-Agent": "darglobal-bot/1.0"},
            timeout=REQUEST_TIMEOUT
        )
        if res.status_code == 200:
            data = res.json()
            results = []

            # Check Abstract
            abstract = data.get("AbstractText", "").strip()
            abstract_url = data.get("AbstractURL", "").strip()
            heading = data.get("Heading", "DuckDuckGo Topic").strip()

            if abstract and abstract_url:
                results.append({
                    "title": heading or clean_query,
                    "url": abstract_url,
                    "snippet": abstract,
                    "source_type": "external"
                })

            # Check Related Topics
            for topic in data.get("RelatedTopics", []):
                if len(results) >= max_results:
                    break
                if isinstance(topic, dict) and "Text" in topic and "FirstURL" in topic:
                    text = topic.get("Text", "").strip()
                    url = topic.get("FirstURL", "").strip()
                    if text and url:
                        results.append({
                            "title": text.split(" - ")[0] if " - " in text else clean_query,
                            "url": url,
                            "snippet": text,
                            "source_type": "external"
                        })

            if results:
                return results
    except Exception as e:
        logger.warning(f"DuckDuckGo API search failed: {e}. Falling back...")

    # 4. Wikipedia Open Search API (Free, high-quality encyclopedia knowledge)
    try:
        logger.info("Querying Wikipedia API...")
        res = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": clean_query,
                "format": "json",
                "utf8": 1,
                "srlimit": max_results
            },
            headers={"User-Agent": "darglobal-bot/1.0 (info@darglobal.example)"},
            timeout=REQUEST_TIMEOUT
        )
        if res.status_code == 200:
            data = res.json()
            search_items = data.get("query", {}).get("search", [])
            results = []
            for item in search_items:
                title = item.get("title", "")
                snippet_raw = item.get("snippet", "")
                # Clean html tags from snippet
                clean_snippet = re.sub(r"<[^>]+>", "", snippet_raw)
                page_id = item.get("pageid", "")
                url = f"https://en.wikipedia.org/?curid={page_id}" if page_id else f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                if clean_snippet and title:
                    results.append({
                        "title": title,
                        "url": url,
                        "snippet": clean_snippet,
                        "source_type": "external"
                    })
            if results:
                return results
    except Exception as e:
        logger.warning(f"Wikipedia search failed: {e}")

    return []

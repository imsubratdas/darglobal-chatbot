"""
rag.py - Hybrid Search RAG Pipeline for darglobal-bot
Combines BM25 lexical search and dense semantic vector search via Reciprocal Rank Fusion (RRF).
"""

import os
import re
import json
import logging
import hashlib
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from fastembed import TextEmbedding
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

logger = logging.getLogger("darglobal_bot.rag")

PACKAGE_DATA_DIR = BASE_DIR / "data"
DATA_DIR_SETTING = Path(os.getenv("DATA_DIR", PACKAGE_DATA_DIR))
DATA_DIR = (DATA_DIR_SETTING if DATA_DIR_SETTING.is_absolute() else BASE_DIR / DATA_DIR_SETTING).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
PAGES_FILE = DATA_DIR / "pages.json"
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)
CHUNKS_CACHE_FILE = CACHE_DIR / "chunks.json"
EMBEDDINGS_CACHE_FILE = CACHE_DIR / "embeddings.npy"

# On a fresh persistent deployment volume, seed the scraped corpus and vector cache
# from the copy shipped in the image. Existing persistent files always take priority.
if DATA_DIR != PACKAGE_DATA_DIR.resolve():
    packaged_pages = PACKAGE_DATA_DIR / "pages.json"
    if not PAGES_FILE.exists() and packaged_pages.exists():
        shutil.copy2(packaged_pages, PAGES_FILE)
    packaged_cache = PACKAGE_DATA_DIR / "cache"
    if packaged_cache.exists():
        for filename in ("chunks.json", "embeddings.npy"):
            source, destination = packaged_cache / filename, CACHE_DIR / filename
            if source.exists() and not destination.exists():
                shutil.copy2(source, destination)

# Standard English stopwords for BM25 tokenization
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me",
    "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd",
    "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


def tokenize(text: str) -> List[str]:
    """Tokenize and normalize text for BM25 lexical matching."""
    text = text.lower()
    # Replace non-alphanumeric with spaces, keep alphanumeric and hyphens inside words
    tokens = re.findall(r"\b[a-z0-9_-]+\b", text)
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


def split_text_into_chunks(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """Split text into overlapping chunks, attempting sentence boundaries where possible."""
    chunks = []
    text = text.strip()
    if not text:
        return chunks

    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        # If not at end, try to break at paragraph or period
        if end < text_len:
            period_pos = text.rfind(".", start + chunk_size // 2, end)
            newline_pos = text.rfind("\n", start + chunk_size // 2, end)
            break_pos = max(period_pos, newline_pos)
            if break_pos != -1:
                end = break_pos + 1

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= text_len:
            break
        start = max(end - overlap, start + 1)

    return chunks


class HybridRAG:
    """
    Production-grade Hybrid RAG combining:
    1. BM25 lexical keyword retrieval (exact project names, cities, amenities, terminology)
    2. Dense semantic vector retrieval via FastEmbed (meaning, intent, synonyms)
    3. Reciprocal Rank Fusion (RRF) for balanced multi-retrieval ranking
    4. Metadata preservation and context sufficiency estimation
    """

    def __init__(self, data_file: Path = PAGES_FILE):
        self.data_file = data_file
        self.chunks: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.embedding_model: Optional[TextEmbedding] = None
        self.chunk_embeddings: Optional[np.ndarray] = None
        self._transformer_unavailable = False
        self._lsa_vectorizer = None
        self._lsa_model = None
        self._lsa_embeddings = None
        self._load_and_index()

    def _load_and_index(self):
        """Loads pages, creates chunks, computes or loads embeddings, and builds BM25 index."""
        if not self.data_file.exists():
            logger.warning(f"Data file not found at {self.data_file}")
            return

        with open(self.data_file, "r", encoding="utf-8") as f:
            pages = json.load(f)
        pages_signature = hashlib.sha256(self.data_file.read_bytes()).hexdigest()

        logger.info(f"Loaded {len(pages)} pages from {self.data_file}")

        cached_data = {}
        try:
            if CHUNKS_CACHE_FILE.exists():
                with open(CHUNKS_CACHE_FILE, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
        except Exception as exc:
            logger.warning("Failed loading chunk cache; rebuilding: %s", exc)

        if (cached_data.get("page_count") == len(pages)
                and cached_data.get("pages_signature") == pages_signature):
            self.chunks = cached_data.get("chunks", [])
            if EMBEDDINGS_CACHE_FILE.exists():
                try:
                    vectors = np.load(EMBEDDINGS_CACHE_FILE, mmap_mode="r")
                    if len(vectors) == len(self.chunks):
                        self.chunk_embeddings = np.asarray(vectors, dtype=np.float32)
                except Exception as exc:
                    logger.warning("Failed loading vector cache: %s", exc)
        else:
            # Rebuild chunks, then keep old vectors only if every cached chunk still aligns.
            old_chunks = cached_data.get("chunks", [])
            old_vectors = None
            if EMBEDDINGS_CACHE_FILE.exists():
                try:
                    old_vectors = np.load(EMBEDDINGS_CACHE_FILE, mmap_mode="r")
                except Exception:
                    pass
            self._build_chunks(pages)
            if (old_vectors is not None and len(old_vectors) == len(self.chunks)
                    and len(old_chunks) == len(self.chunks)
                    and all(old.get("search_text") == new.get("search_text") and old.get("url") == new.get("url")
                            for old, new in zip(old_chunks, self.chunks))):
                self.chunk_embeddings = np.asarray(old_vectors, dtype=np.float32)
            self._save_chunk_cache(len(pages), pages_signature)

        if (self.chunk_embeddings is None
                and os.getenv("BUILD_EMBEDDINGS_ON_STARTUP", "false").lower() in ("true", "1", "yes")):
            self._build_embeddings()

        # Build BM25 index
        self._build_bm25()

    def _build_chunks(self, pages: List[Dict[str, Any]]):
        """Build chunks with rich metadata."""
        self.chunks = []
        chunk_id = 0
        for page in pages:
            text = page.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue

            company = page.get("company", "Unknown")
            url = page.get("url", "")
            title = page.get("title", "")
            description = page.get("description", "")

            pieces = split_text_into_chunks(text, chunk_size=800, overlap=150)
            for idx, piece in enumerate(pieces):
                self.chunks.append({
                    "id": chunk_id,
                    "company": company,
                    "url": url,
                    "title": title,
                    "description": description,
                    "text": piece,
                    "chunk_index": idx,
                    # Combined representation for BM25 and embedding
                    "search_text": f"{company} | {title} | {piece}"
                })
                chunk_id += 1

        logger.info(f"Generated {len(self.chunks)} chunks across {len(pages)} pages")

    def _build_embeddings(self):
        """Generate dense embeddings using FastEmbed and cache to disk."""
        if not self.chunks:
            return

        logger.info("Initializing FastEmbed (BAAI/bge-small-en-v1.5)...")
        try:
            if self.embedding_model is None:
                self.embedding_model = TextEmbedding(
                    model_name="BAAI/bge-small-en-v1.5",
                    local_files_only=os.getenv("FASTEMBED_LOCAL_ONLY", "false").lower() in ("true", "1", "yes")
                )

            texts = [c["search_text"] for c in self.chunks]
            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            embeddings_list = list(self.embedding_model.embed(texts, batch_size=64))
            self.chunk_embeddings = np.array(embeddings_list, dtype=np.float32)
        except Exception as exc:
            logger.warning("Semantic model is unavailable; continuing with BM25 retrieval: %s", exc)
            self._transformer_unavailable = True
            self.chunk_embeddings = None
            return

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(self.chunk_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.chunk_embeddings = self.chunk_embeddings / norms

        # Save cache
        try:
            with open(CHUNKS_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "page_count": len(json.loads(self.data_file.read_text(encoding="utf-8"))),
                    "pages_signature": hashlib.sha256(self.data_file.read_bytes()).hexdigest(),
                    "chunks": self.chunks
                }, f)
            np.save(EMBEDDINGS_CACHE_FILE, self.chunk_embeddings)
            logger.info("Saved embeddings and chunks cache to disk")
        except Exception as e:
            logger.warning(f"Could not write cache: {e}")

    def _build_bm25(self):
        """Build BM25 index on tokenized chunk search text."""
        if not self.chunks:
            return

        corpus_tokens = [tokenize(c["search_text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(corpus_tokens, k1=1.5, b=0.75)
        logger.info("BM25 index built successfully")

    def _save_chunk_cache(self, page_count: int, pages_signature: str):
        try:
            with open(CHUNKS_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "page_count": page_count,
                    "pages_signature": pages_signature,
                    "chunks": self.chunks
                }, f)
        except Exception as exc:
            logger.warning("Could not save chunk cache: %s", exc)

    def _search_bm25(self, query: str, top_k: int = 25) -> List[Tuple[int, float]]:
        """Perform BM25 lexical keyword search."""
        if self.bm25 is None or not self.chunks:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0.05:
                results.append((int(idx), score))
        return results

    def _search_semantic(self, query: str, top_k: int = 25) -> List[Tuple[int, float]]:
        """Perform dense vector semantic search using FastEmbed."""
        if not self.chunks:
            return []

        if not self._transformer_unavailable and self.chunk_embeddings is not None:
            try:
                if self.embedding_model is None:
                    self.embedding_model = TextEmbedding(
                        model_name="BAAI/bge-small-en-v1.5",
                        local_files_only=os.getenv("FASTEMBED_LOCAL_ONLY", "false").lower() in ("true", "1", "yes")
                    )
                query_embed = list(self.embedding_model.embed([query]))[0]
                query_vec = np.array(query_embed, dtype=np.float32)
                norm = np.linalg.norm(query_vec)
                if norm > 0:
                    query_vec = query_vec / norm
                scores = np.dot(self.chunk_embeddings, query_vec)
                top_indices = np.argsort(scores)[::-1][:top_k]
                return [(int(idx), float(scores[idx])) for idx in top_indices]
            except Exception as exc:
                logger.warning("FastEmbed query failed; enabling local LSA vectors: %s", exc)
                self._transformer_unavailable = True

        # Offline fallback: dense latent-semantic vectors built from the local corpus.
        try:
            if self._lsa_vectorizer is None:
                corpus = [c["search_text"] for c in self.chunks]
                self._lsa_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=30000, stop_words="english")
                sparse = self._lsa_vectorizer.fit_transform(corpus)
                dimensions = min(128, sparse.shape[0] - 1, sparse.shape[1] - 1)
                if dimensions < 2:
                    self._lsa_embeddings = normalize(sparse).tocsr()
                else:
                    self._lsa_model = TruncatedSVD(n_components=dimensions, n_iter=5, random_state=17)
                    self._lsa_embeddings = normalize(self._lsa_model.fit_transform(sparse)).astype(np.float32)
            query_sparse = self._lsa_vectorizer.transform([query])
            if self._lsa_model is not None:
                query_vector = normalize(self._lsa_model.transform(query_sparse))[0]
                scores = self._lsa_embeddings @ query_vector
            else:
                scores = (self._lsa_embeddings @ query_sparse.T).toarray().ravel()
            top_indices = np.argsort(scores)[::-1][:top_k]
            return [(int(idx), float(scores[idx])) for idx in top_indices]
        except Exception as exc:
            logger.warning("Local semantic vector search failed: %s", exc)
            return []

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        semantic_weight: float = 1.0
    ) -> Dict[str, Any]:
        """
        Execute Hybrid Search combining BM25 and Semantic Retrieval via Reciprocal Rank Fusion.

        Returns:
            Dict containing:
            - 'results': List of ranked chunk dicts with source metadata and scores
            - 'sufficient': bool, whether internal knowledge has sufficient relevance
            - 'best_semantic_score': float
            - 'best_bm25_score': float
        """
        if not self.chunks:
            return {
                "results": [],
                "sufficient": False,
                "best_semantic_score": 0.0,
                "best_bm25_score": 0.0
            }

        bm25_hits = self._search_bm25(query, top_k=30)
        semantic_hits = self._search_semantic(query, top_k=30)

        best_bm25 = bm25_hits[0][1] if bm25_hits else 0.0
        best_semantic = semantic_hits[0][1] if semantic_hits else 0.0

        # Reciprocal Rank Fusion (RRF)
        # RRF_score(d) = sum( w_m / (k + rank_m(d)) )
        rrf_scores: Dict[int, float] = {}
        bm25_rank_map: Dict[int, int] = {}
        sem_rank_map: Dict[int, int] = {}
        bm25_score_map: Dict[int, float] = {}
        sem_score_map: Dict[int, float] = {}

        for rank, (idx, score) in enumerate(bm25_hits):
            bm25_rank_map[idx] = rank + 1
            bm25_score_map[idx] = score
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (bm25_weight / (rrf_k + rank + 1))

        for rank, (idx, score) in enumerate(semantic_hits):
            sem_rank_map[idx] = rank + 1
            sem_score_map[idx] = score
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (semantic_weight / (rrf_k + rank + 1))

        # Sort candidate chunks by RRF score
        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)

        # Deduplicate / Diversity filter: avoid taking too many chunks from the same URL
        final_results = []
        url_count: Dict[str, int] = {}

        for idx in sorted_indices:
            chunk = self.chunks[idx].copy()
            url = chunk.get("url", "")

            # Allow max 2 chunks per unique URL to ensure diversity across pages
            if url_count.get(url, 0) >= 2:
                continue

            url_count[url] = url_count.get(url, 0) + 1
            chunk["rrf_score"] = float(rrf_scores[idx])
            chunk["bm25_rank"] = bm25_rank_map.get(idx, None)
            chunk["bm25_score"] = bm25_score_map.get(idx, 0.0)
            chunk["semantic_rank"] = sem_rank_map.get(idx, None)
            chunk["semantic_score"] = sem_score_map.get(idx, 0.0)

            final_results.append(chunk)
            if len(final_results) >= top_k:
                break

        # Sufficiency evaluation:
        # If semantic score is above 0.40 OR BM25 score is strong (> 2.0), consider sufficient
        sufficient = False
        if best_semantic >= 0.45 or best_bm25 >= 2.5:
            sufficient = True
        elif best_semantic >= 0.38 and best_bm25 >= 1.0:
            sufficient = True

        return {
            "results": final_results,
            "sufficient": sufficient,
            "best_semantic_score": float(best_semantic),
            "best_bm25_score": float(best_bm25)
        }


# Singleton instance
_rag_instance: Optional[HybridRAG] = None

def get_rag() -> HybridRAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = HybridRAG()
    return _rag_instance

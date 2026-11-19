# darglobal-bot

An independent AI informational assistant for exploring collected DarGlobal and Wasalt website information. It combines lexical and semantic retrieval, keeps conversations in SQLite, and shows source links with internal or external attribution.

> This is an independent assistant and is not an official DarGlobal or Wasalt product.

## Features and architecture

- FastAPI serves the API and the responsive single-page HTML interface.
- SQLAlchemy and SQLite persist accounts, owned chats, messages, pins, and timestamps. The existing `data/propintel.db` file is retained for compatibility and existing records.
- Passwords use `pwdlib`'s recommended password hash. Session JWTs are held in an HTTP-only cookie.
- `data/pages.json` is the scraped knowledge base. `rag.py` chunks it while retaining company, title, URL, and chunk index.
- BM25 (`rank-bm25`) finds exact names and terms; FastEmbed's `BAAI/bge-small-en-v1.5` model produces dense vectors. The two ranked lists are merged with Reciprocal Rank Fusion (RRF), then capped to a small context with a maximum of two chunks per source URL. There is no separate reranker.
- OpenRouter provides optional answer generation. By default the app uses its current `openrouter/free` router. The response parser accepts text and text-block content, rejects empty responses, and never treats reasoning traces as answers.
- If internal retrieval is judged insufficient, optional external search tries Tavily, then SerpAPI, then DuckDuckGo Instant Answers and Wikipedia. Search is bounded by provider timeouts and can be disabled. No results are invented. External sources are labeled in the chat.

## Requirements

- Python 3.12 (the pinned NumPy version requires Python 3.12)
- Internet access on first startup if the FastEmbed model is not already cached
- OpenRouter API key for generated answers. Without it, the backend returns a non-empty evidence summary or a clear service-unavailable message.
- Optional Tavily or SerpAPI key for broader external web search. Tavily currently offers a free Researcher plan with 1,000 monthly credits and no credit card; check its [current pricing](https://www.tavily.com/pricing). DuckDuckGo Instant Answers and Wikipedia are best-effort fallbacks, not substitutes for full web search.

## Windows local setup

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env`: set `SECRET_KEY` to a long random secret and, for generated AI answers, set `OPENROUTER_API_KEY`. Add `TAVILY_API_KEY` for the recommended external search provider. Never commit `.env`.

Start the app:

```powershell
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Visit <http://127.0.0.1:8000>. The API docs are at <http://127.0.0.1:8000/docs>; health status is at `/health`.

## Environment variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `SECRET_KEY` | Production | Signs session JWTs. If absent, a random process-local key is generated and sessions reset on restart. |
| `COOKIE_SECURE` | HTTPS deployments | Set `true` to require HTTPS for the session cookie. |
| `OPENROUTER_API_KEY` | No | Enables generated responses. |
| `OPENROUTER_MODELS` | No | Comma-separated OpenRouter model IDs; defaults to `openrouter/free`. |
| `FASTEMBED_LOCAL_ONLY` | No | When true, avoids downloading the embedding model and uses local latent semantic vectors if it is not cached. |
| `ENABLE_EXTERNAL_SEARCH` | No | Set `false` to disable external search; defaults to enabled. |
| `TAVILY_API_KEY` | No | Tavily search; tried first when configured. |
| `SERPAPI_API_KEY` | No | SerpAPI fallback when configured. |
| `DATABASE_URL` | No | SQLAlchemy URL. Defaults to `data/propintel.db` beside the application. |
| `DATA_DIR` | No | Directory for the SQLite database, scraped pages, and embedding cache. Defaults to `data/`; deployments with a mounted disk should point it at that disk. |
| `LEGACY_DATABASE_FILE` | No | Existing SQLite file to copy into a new persistent data directory on first startup. Keep it private and outside the Docker image. |

## API endpoints

- `POST /api/register`, `POST /api/login`, `POST /api/logout`, `GET /api/me`
- `GET /api/chats`, `POST /api/chats`, `GET /api/chats/{chat_id}`
- `PATCH /api/chats/{chat_id}` (rename and/or pin state), `POST /api/chats/{chat_id}/pin`, `DELETE /api/chats/{chat_id}`
- `POST /api/chats/{chat_id}/messages` (persistent authenticated chat turn)
- `POST /chat` (standalone RAG request; accepts optional history and an owned chat ID)
- `GET /health`

Chat answers use `answer`, `sources`, `answer_type`, `external_sources`, and `retrieval_info`. `answer_type` is `internal`, `external`, `mixed`, or `general`. A general answer has no collected or external source evidence.

## Scraping and data

Run `python scraper.py` from the project root to refresh `data/pages.json`. The scraper respects robots.txt and pacing controls defined in the script. Review the target sites' current terms and crawl policies before refreshing. Existing scraped pages and SQLite records are not deleted during normal startup. The embedding cache in `data/cache/` is checked against a content signature of `pages.json` and rebuilt when the source data changes.

## Docker

```sh
docker build -t darglobal-bot .
docker run --rm -p 8000:8000 \
  -e SECRET_KEY='replace-with-a-long-random-secret' \
  -e COOKIE_SECURE=false \
  -e OPENROUTER_API_KEY='your-key' \
  -e DATA_DIR=/var/data \
  -v darglobal-bot-data:/var/data \
  darglobal-bot
```

Mount `/var/data` as a persistent volume to preserve the SQLite database, scraped corpus, and retrieval cache. On first start, the application seeds a fresh volume with the corpus and cache shipped in the image; later data in the volume takes precedence. The existing local SQLite file is excluded from the image because it can contain account data. To migrate its records, place it privately on the persistent volume and set `LEGACY_DATABASE_FILE=/var/data/legacy-propintel.db` before the first start. The image exposes port 8000 and honors the `PORT` environment variable. The health check calls `/health`.

## Deployment

`render.yaml` configures a Docker web service with a persistent disk mounted at `/var/data`, generated `SECRET_KEY`, HTTPS cookies, and secret placeholders. Connect the repository to Render, provide `OPENROUTER_API_KEY` and optionally `TAVILY_API_KEY` in its environment settings, then deploy. If migrating the current local users, privately place the SQLite file on the persistent disk and set `LEGACY_DATABASE_FILE` before the first start. This repository has not been deployed from this workspace.

## Limitations

- Retrieval quality depends on the coverage and freshness of `data/pages.json`; it is not a live listings feed.
- External search depends on provider availability. DuckDuckGo Instant Answers and Wikipedia provide limited coverage when API keys are absent.
- Free model routing is rate limited and may vary in availability. The app uses an evidence summary fallback when model generation fails.
- SQLite is suitable for a small deployment. A larger multi-instance service should use a managed database and shared storage for the embedding index.

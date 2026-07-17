# RAG Docs API

A production-shaped **Retrieval-Augmented Generation** service. Upload documents over a REST API, get back grounded, cited answers from an LLM.

Built with FastAPI, ChromaDB, MongoDB, Redis, PostgreSQL and Docker. Runs end-to-end with **no API key and no cloud spend** — the default LLM backend is a local Ollama model, and embeddings run on CPU via ONNX Runtime with no torch dependency.

[![CI](https://github.com/eaibrahim32/rag-docs-api/actions/workflows/ci.yml/badge.svg)](https://github.com/eaibrahim32/rag-docs-api/actions)

---

## Why this exists

Most RAG demos are a notebook that calls OpenAI in a loop. This is the other version: the one that has to stay up. It has an async ingestion path, health probes, a cache, typed errors, structured logs, 52 tests and a CI pipeline — because that's what separates "I built a RAG demo" from "I maintain a RAG service in production."

---

## Architecture

```
                        ┌──────────────┐
   POST /documents ────▶│              │───▶ BackgroundTask ──▶ ingestion pipeline
   POST /query     ────▶│   FastAPI    │                              │
   GET  /search    ────▶│              │◀── hybrid retrieval ◀────────┘
                        └──────┬───────┘
                               │
        ┌──────────────┬───────┴───────┬──────────────┐
        ▼              ▼               ▼              ▼
   ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
   │PostgreSQL│  │ MongoDB  │   │  Chroma  │   │  Redis   │
   │ metadata │  │  chunk   │   │ vectors  │   │  answer  │
   │ + status │  │  bodies  │   │  (HNSW)  │   │  cache   │
   └─────────┘   └──────────┘   └──────────┘   └──────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │ Ollama / OpenAI │
                              └─────────────────┘
```

**Ingestion:** `bytes → text → chunks → (Mongo bodies + Chroma vectors) → SQL status`
**Retrieval:** `question → vector search ∥ keyword search → RRF fusion → hydrate → LLM → cited answer`

### Why four datastores instead of one

Each one is doing a job the others do badly.

| Store | Holds | Why not somewhere else |
|---|---|---|
| **PostgreSQL** | Document metadata, ingest status, chunk counts | Needs filtering, sorting and pagination — relational work |
| **MongoDB** | Raw text and chunk bodies + text index | Schemaless blobs read by key; also powers the keyword leg of hybrid search |
| **Chroma** | 384-dim vectors + resolver metadata | Purpose-built ANN index; chunk text is *not* duplicated here |
| **Redis** | Answer cache, rate limiting | A repeated question should never re-run embedding + inference |

---

## Design decisions worth defending

**Hybrid retrieval with Reciprocal Rank Fusion.** Dense vectors handle paraphrase (*"how do I cancel"* → *"termination clause"*) but reliably miss exact identifiers — error codes, SKUs, names. A keyword index nails those and misses paraphrase. RRF merges both rankings using rank position rather than score, so the two legs don't need comparable score scales. Either leg can fail and retrieval degrades instead of dying.

**Upload returns 202, not 200.** Embedding a 200-page PDF takes tens of seconds. Holding an HTTP connection open for that is a timeout waiting to happen. Ingestion runs as a background task; clients poll `GET /documents/{id}` for `pending → processing → ready | failed`. Failures are recorded with the error, not swallowed.

**Deduplication by SHA-256.** Re-uploading identical bytes returns the existing document instead of re-embedding it. Ingestion is the expensive operation; making it idempotent makes retries free.

**The cache degrades, it doesn't fail.** Every Redis call is wrapped — a cache outage means slow answers, not a 500. The rate limiter fails open for the same reason. Ingest and delete invalidate answer keys, because the retrieval scope just changed and cached answers are stale.

**Two LLM backends, one interface.** Ollama by default (no key, no spend, runs offline); OpenAI is a config change, not a code change. Both share a prompt that forbids outside knowledge and requires inline citations, so hallucination is bounded by retrieval.

**Embeddings on ONNX Runtime, not torch.** The default backend runs all-MiniLM-L6-v2 through onnxruntime, which chromadb already depends on. The sentence-transformers build of the *same model* pulls torch, and torch pulls ~2.5 GB of CUDA wheels — cuBLAS, cuDNN, NCCL, Triton — none of which execute on a CPU host. Identical vectors, ~250 MB of install instead of ~3 GB, and a Docker image around 1 GB instead of 5. sentence-transformers stays available behind a config flag for GPU hosts.

**`/search` exists separately from `/query`.** It returns ranked chunks with no LLM call. Most RAG bugs are retrieval bugs wearing a generation costume — this endpoint lets you see which one you actually have.

**`/health` vs `/ready`.** Liveness never touches dependencies; readiness checks all four stores in parallel with a timeout and returns 503 if any is down. That's the distinction an orchestrator needs to route traffic correctly.

---

## Quickstart

Open in a Codespace (a devcontainer is committed — Docker-in-Docker, 4 cores), or clone locally:

```bash
git clone https://github.com/eaibrahim32/rag-docs-api.git
cd rag-docs-api
make up            # builds the API, starts Postgres + Mongo + Redis + Ollama, pulls the model
```

**Codespaces note:** the free quota is 120 core-hours (= 60h on 2-core, 30h on 4-core) and
15 GB-month of storage that accrues until the codespace is *deleted*, not merely stopped.
Use 2-core for `make test` / `make lint`; only pick 4-core when you actually run the stack,
and delete the codespace afterwards. Full CI runs on GitHub Actions, which is free for
public repos — so the test suite is the proof, not a long-lived codespace.

**Resource note:** `make up` defaults to `llama3.2:1b`, which fits a 4-core/8GB machine.
`llama3.1:8b` needs roughly 8GB for the model alone — use `make up MODEL=llama3.1:8b` only
where there's headroom, or set `LLM_BACKEND=openai` and skip local inference entirely.
`make up-nollm` starts everything except Ollama: `/search` works, `/query` needs an LLM.

Open http://localhost:8000/docs for interactive OpenAPI docs.

```bash
# ingest
curl -F "file=@handbook.pdf" http://localhost:8000/api/v1/documents
# {"id":"a3f...","status":"pending",...}

# poll until ready
curl http://localhost:8000/api/v1/documents/a3f...

# ask
curl -X POST http://localhost:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the notice period for resignation?"}'
```

```json
{
  "answer": "The notice period is 30 days for permanent employees [1], extended to 60 days for management grades [2].",
  "citations": [
    {"document_id": "a3f...", "filename": "handbook.pdf", "chunk_index": 12, "score": 0.031, "snippet": "..."}
  ],
  "cached": false,
  "latency_ms": 1840,
  "model": "ollama/llama3.1:8b"
}
```

### Local development (no Docker)

```bash
make install
make test          # 52 tests, no Docker, no network, no model download
make run
```

---

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/documents` | Upload + ingest (202, async) |
| `GET` | `/api/v1/documents` | List with pagination + status filter |
| `GET` | `/api/v1/documents/{id}` | Ingestion status and chunk count |
| `DELETE` | `/api/v1/documents/{id}` | Purge from all stores |
| `POST` | `/api/v1/query` | Cited answer over the corpus |
| `GET` | `/api/v1/search` | Ranked chunks, no LLM — retrieval debugging |
| `GET` | `/health` | Liveness |
| `GET` | `/ready` | Readiness — checks all four stores |

Supported formats: PDF, DOCX, TXT, MD, HTML, CSV.

---

## Testing

Two layers, and the split is deliberate.

```bash
make test              # 52 unit tests — no Docker, no network, ~14s
make services          # start Postgres + Mongo + Redis
make test-integration  # 47 integration tests against those live services
```

**Unit** (`tests/unit`) fakes Mongo, Redis and the LLM and swaps in a deterministic
hashing embedder, so it runs anywhere in seconds with no Docker, no network and no
model download. It covers what breaks quietly: chunk boundaries and overlap, RRF
ranking, cache-key normalisation, HTTP error contracts, and a full
upload → ingest → index → retrieve → delete round trip.

**Integration** (`tests/integration`) fakes nothing. Real PostgreSQL, real MongoDB
(including the text index behind hybrid search's keyword leg), real Redis, real ONNX
embeddings, and the real httpx client driven against a stub server that speaks
Ollama's wire format. It asserts things SQLite and fakes will happily let past:
timezone-aware timestamps, Mongo text-score ranking, cache TTL expiry and
invalidation-on-ingest, and that a cache hit never reaches the model at all.

Both run on every push. CI is free on public repos, so the integration job is the
proof — not a screenshot of a stack someone once ran locally.

---

## Engineering practices

- **Structured JSON logging** with per-request correlation IDs propagated via `x-request-id`
- **Typed errors** (`AppError` subclasses) mapped to HTTP status + stable machine-readable codes by one handler
- **Multi-stage Docker build** — build toolchain stays out of the runtime image; runs as non-root with a healthcheck
- **Compose healthchecks + `depends_on: condition: service_healthy`** so the API doesn't race its databases on boot
- **CI**: ruff → unit tests on Python 3.11 and 3.12 → integration tests against live Postgres/Mongo/Redis service containers → Docker build with layer caching
- **12-factor config** — every value environment-overridable via pydantic-settings, no secrets in the repo

---

## Stack

Python 3.11 · FastAPI · Pydantic v2 · SQLAlchemy 2.0 (async) · PostgreSQL · MongoDB (Motor) · Redis · ChromaDB · ONNX Runtime (all-MiniLM-L6-v2) · Ollama / OpenAI · Docker Compose · pytest · ruff · GitHub Actions

## License

MIT

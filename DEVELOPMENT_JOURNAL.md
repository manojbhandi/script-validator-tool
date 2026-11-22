# Development Journal — Creative Script Validation Service

This is a step-by-step record of how this repo was actually built: every function, why it exists,
what the Python syntax in it means, and the bugs/issues we hit along the way with how they got
fixed. Written so that opening this file later (once the chat that built it is gone) still explains
the "why" behind the code, not just the "what" — the code already tells you the what.

Read it top to bottom in build order, or jump to a section using the headings.

---

## 0. Starting point and approach

The assignment: given a campaign brief and a script written against it, score the script on three
axes (brief alignment, marketing message quality, product claim validity) using RAG against TFS
product manuals, host it on Cloud Run/Lambda, and show retrieval accuracy after every run.

Two decisions made before writing any code:

1. **No LangChain/LlamaIndex.** The corpus is small (a few hundred chunks), so hand-rolling
   ingestion/retrieval with direct API calls is less code to understand than learning a framework,
   and it's a better demonstration of actually understanding RAG rather than calling a library.
2. **Build in vertical slices, not horizontal layers.** Originally the plan was "define all Pydantic
   schemas first, then all clients, then all pipeline stages." That got abandoned early in favour of
   "get ingestion working end-to-end against real data, then retrieval end-to-end, then scoring
   end-to-end" — adding each schema right before the function that needs it. This surfaces real
   problems (bad PDF text, rate limits) immediately instead of after a week of paper design.

---

## 1. `app/config.py` — typed settings

**What it does:** one `Settings` object, read once from `.env`, that every other file imports
instead of calling `os.getenv` directly.

```python
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    llm_provider: str = "gemini"
    llm_api_key: str = ""
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_model: str = "gemini-3.6-flash"
    ...
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

**Why this shape:**
- `pydantic-settings`' `BaseSettings` maps env var names to field names case-insensitively
  (`GLM_API_KEY` env var ↔ `glm_api_key` field, later renamed `llm_api_key`), and it **validates
  types on boot** — if `SIMILARITY_THRESHOLD=0.6s` is in `.env`, the app refuses to start with a
  clear Pydantic error instead of crashing three requests later inside a comparison. This exact
  thing happened (see §12) and the fail-fast behaviour is what caught it immediately.
- `@lru_cache` on `get_settings()` means `.env` is read once and every caller gets the same object,
  not a fresh file read per call.

**Python syntax:**
- `field_name: type = default` is a type-annotated class attribute. Pydantic *reads* the type hint
  and enforces/converts it at runtime — plain Python ignores type hints, Pydantic does not.
- `@lru_cache` is a decorator: `@foo` above `def bar(): ...` is shorthand for
  `bar = foo(bar)`. `lru_cache` wraps the function so the first call runs it and remembers the
  result; every later call returns the cached value.
- `extra="ignore"` — unknown keys in `.env` are ignored rather than raising, useful while iterating.

**Bug hit:** none in this file itself, but it later **caught** a bug — a stray `s` typed at the end
of `SIMILARITY_THRESHOLD=0.6s` in `.env` — with a validation error naming the exact field and bad
value, rather than a confusing failure deep in retrieval code.

---

## 2. `app/main.py` — the FastAPI app

```python
from fastapi import FastAPI
from app.routers import ingest, score
from app.utils.logging import configure_logging
from app.config import get_settings

configure_logging()
app = FastAPI(title="Creative Script Validation Service")
app.include_router(ingest.router)
app.include_router(score.router)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", include_in_schema=False)
def index():
    return FileResponse("app/static/index.html")

@app.get("/health")
def health():
    settings = get_settings()
    return {"status": "ok", "llm_provider": settings.llm_provider, "embedding_provider": settings.embedding_provider}
```

**Bug hit — the very first one of the project:** the initial `/health` route was written as

```python
app.get("/health")   # missing @
def health(): ...
```

Without the `@`, that line just **calls** `app.get("/health")` (which returns a decorator) and
throws the result away — `health()` is never registered with FastAPI. Result: `GET /health`
returned 404. The fix was adding the `@`. This is the single most instructive early bug: a
decorator with no `@` in front of it isn't a decorator, it's just a function call whose return
value goes nowhere.

**Second bug** (much later, once the frontend was added): the mount and `/` route were written out
in chat but never actually landed in the file — `git diff` showed the imports were added but the
`app.mount(...)` and `@app.get("/")` lines were missing, so `GET /` was 404 even after `GET /health`
worked. Lesson: after any multi-line instruction, re-read the file rather than assuming every line
landed.

**Python syntax:**
- `@app.get("/health")` is a decorator that registers the function below it as the handler for
  `GET /health`. `app.include_router(...)` merges routes defined in another file (`APIRouter`) into
  the main app — this is how `/score` and `/ingest` live in separate files but still work.
- `StaticFiles`/`FileResponse` are FastAPI/Starlette helpers for serving files instead of JSON.

---

## 3. `app/models/schemas.py` — the typed contract

Every function in the pipeline takes and returns one of these Pydantic models instead of a raw
`dict`. Built incrementally, one group of classes per pipeline stage, not all at once.

### Enums

```python
class SectionType(str, Enum):
    SPEC = "spec"; CLAIM = "claim"; WARNING = "warning"; USAGE = "usage"; WARRANTY = "warranty"; OTHER = "other"

class ClaimVerdict(str, Enum):
    SUPPORTED = "supported"; UNSUPPORTED = "unsupported"; CONTRADICTED = "contradicted"; UNVERIFIABLE = "unverifiable"

class RetrievalStatus(str, Enum):
    WELL_GROUNDED = "well_grounded"; LOW_CONFIDENCE = "low_confidence"
```

**Why `(str, Enum)` and not a plain `Enum`:** inheriting from both `str` and `Enum` means each
member *is* a string — it serialises to JSON as `"spec"` rather than `SectionType.SPEC` — **and**
Pydantic rejects any value outside the fixed set. If an LLM ever returns `"verdict": "probably_true"`
instead of one of the four allowed verdicts, validation fails loudly right there instead of storing
garbage that breaks something three functions downstream.

### Ingestion shapes: `RawPage → Chunk → EmbeddedChunk → ScoredChunk`

Each stage of the ingestion pipeline adds exactly one thing to the previous shape:

```python
class RawPage(BaseModel):
    product_name: str; page_number: int; text: str

class Chunk(BaseModel):
    chunk_id: str; product_name: str; section_type: SectionType = SectionType.OTHER
    page_number: int; text: str; content_hash: str

class EmbeddedChunk(Chunk):        # inherits every Chunk field, adds one
    vector: List[float]

class ScoredChunk(BaseModel):      # like Chunk but with similarity instead of a vector
    chunk_id: str; product_name: str; section_type: SectionType; page_number: int
    text: str; similarity: float = Field(ge=-1.0, le=1.0)
```

**Python syntax:**
- `class EmbeddedChunk(Chunk):` is **inheritance** — `EmbeddedChunk` has every field `Chunk` has
  plus `vector`, without retyping six field names.
- `Field(ge=-1.0, le=1.0)` attaches a **constraint**: cosine similarity is mathematically bounded to
  [-1, 1], so if a bug ever produced 1.7 it fails right at the model boundary instead of leaking into
  a scorecard shown to a user.

### Live-pipeline shapes: `Claim`, `StructuredBrief`, score blocks, `Scorecard`

Added later, right before `claim_extractor.py`/`brief_parser.py`/`scoring.py` needed them:

```python
class Claim(BaseModel):
    claim_id: str; text: str; product_reference: Optional[str] = None

class StructuredBrief(BaseModel):
    target_audience: str; key_message: str; mandatory_inclusions: List[str] = []; tone: str; cta: str

class BriefAlignmentScore(BaseModel):
    score: int = Field(ge=0, le=10); reasoning: str; missing_mandatory_inclusions: List[str] = []

class ClaimVerdictEntry(BaseModel):
    claim_id: str; verdict: ClaimVerdict; evidence: Optional[str] = None; source_chunk: Optional[str] = None

class ClaimValidityScore(BaseModel):
    score: int = Field(ge=0, le=10); reasoning: str; claim_verdicts: List[ClaimVerdictEntry] = []

class Scorecard(BaseModel):
    run_id: str; overall_score: float; overall_feedback: str
    brief_alignment: BriefAlignmentScore; marketing_message_quality: MessageQualityScore
    product_claim_validity: ClaimValidityScore; claims: List[Claim]
    retrieval_coverage_report: List[CoverageEntry]
    offline_eval_baseline: Optional[OfflineEvalBaseline] = None
```

**Why `score: int = Field(ge=0, le=10)` matters in practice:** this is what makes the retry loop in
`generate_structured` (see §7) meaningful. If the LLM returns `11` or `-1`, Pydantic rejects it and
the client retries with the validation error fed back into the prompt, instead of silently averaging
a nonsense number into the final scorecard.

**Python syntax:**
- `Optional[str] = None` — "a string or `None`, default `None`." On Python 3.9 (this project's
  runtime) you must use `typing.Optional`/`typing.List`, not the `str | None` / `list[str]` syntax
  that only works on 3.10+.
- `mandatory_inclusions: List[str] = []` — a mutable default. In *plain* Python classes this is a
  well-known trap (every instance would share the same list object), but Pydantic copies defaults
  per instance, so it's safe here specifically because it's a Pydantic model.

**Bugs hit in this file (typos, all caught by import/runtime checks, not silent):**
- `SectionType.USAGE = "usagae"` — typo would have meant usage slides never got tagged. Fixed to
  `"usage"`.
- `ScoredChunk.page_numeber` — typo. This one mattered more than it looks: later code constructs a
  `ScoredChunk(page_number=...)` from a DB row, and Pydantic would reject it because the field
  didn't exist under that name. Field names are a contract that must match exactly everywhere they're
  used.
- A stray `'` character pasted into the file caused `SyntaxError: EOL while scanning string literal`
  — caught immediately by `python -c "import ast; ast.parse(...)"` before it wasted time. This became
  a standing habit: after any manual edit, run a quick parse/import check before moving on.

---

## 4. Ingestion pipeline — `app/pipeline/ingestion.py`

This is the "offline, run-once" side: turn PDFs/PPTXs into embedded, searchable chunks in Postgres.

### 4.1 The corpus turned out to be PowerPoint decks, not PDFs

The assignment said "product manuals (PDFs)". The actual Drive folder was 46 files: 39 `.pptx`
training decks, 4 `.pdf`, 3 legacy `.ppt`. This changed the design:
- Added `python-pptx` as a second parser alongside PyMuPDF (`fitz`).
- Slides in these decks are short (median ~500 characters) and each one is a self-contained topic —
  so the chunking strategy became **one slide = one chunk** instead of a sliding-window splitter,
  because a human already did the chunking by putting one idea per slide.
- 3 legacy `.ppt` files can't be read by any clean Python library (would need LibreOffice) — logged
  and skipped, documented as a known limitation rather than solved.
- Some files aren't manuals at all (a brand brochure, a market-strategy deck) — handled via
  `manifest.json` (below) rather than ingesting the whole folder blindly.

### 4.2 `data/manuals/manifest.json` — curated file list

A hand-written JSON array `[{"file": ..., "product_name": ..., "line": ...}, ...]`. **Why a
separate file instead of deriving `product_name` from the filename:** filenames are things like
`2018-11_Dr__Belmeur_Advanced_(Ampoule_added).pptx` — dates, brackets, mojibake. The manifest is the
single place that says "these are the files to ingest, and here is each one's clean product name."
A file not listed in the manifest is simply never opened — no special-case code needed to exclude
the brochure or the `.ppt` files.

Started at 17 products to get the pipeline working end-to-end quickly, later expanded to 42 (every
ingestible file in the Drive folder) once the full pipeline was proven.

### 4.3 `ManifestEntry` — validating the manifest itself

```python
class ManifestEntry(BaseModel):
    file: str
    product_name: str
    line: Optional[str] = None
```

Read via `[ManifestEntry(**item) for item in json.load(f)]` — a **list comprehension** combined
with **dict unpacking** (`**item` turns `{"file": "x", "product_name": "y"}` into the keyword
arguments `file="x", product_name="y"`). If a manifest entry had a typo'd key, this fails
immediately with a clear Pydantic error rather than a `KeyError` deep inside the loop.

### 4.4 `_load_pdf_pages` and `_load_pptx_pages`

```python
def _load_pdf_pages(path: Path, product_name: str) -> List[RawPage]:
    pages: List[RawPage] = []
    with fitz.open(path) as doc:
        for page_index, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                pages.append(RawPage(product_name=product_name, page_number=page_index + 1, text=text))
    return pages

def _load_pptx_pages(path: Path, product_name: str) -> List[RawPage]:
    pages: List[RawPage] = []
    prs = Presentation(path)
    for slide_index, slide in enumerate(prs.slides):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                texts.append(shape.text_frame.text)
            if shape.has_table:
                for row in shape.table.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    texts.append(" | ".join(cells))
        text = "\n".join(texts).strip()
        if text:
            pages.append(RawPage(product_name=product_name, page_number=slide_index + 1, text=text))
    return pages
```

**Why both exist as separate small functions rather than one big branching function:** each format
needs completely different library calls, but both must end up producing the exact same `RawPage`
shape so nothing downstream needs to know or care which format a page came from.

**Python syntax:**
- `with fitz.open(path) as doc:` — a **context manager**. Guarantees the file handle is closed when
  the block ends, even if an exception is raised inside it. Same pattern used later for database
  connections.
- `enumerate(doc)` — gives `(index, item)` pairs; `page_index + 1` converts a 0-based index into a
  human page number.
- `if not text: continue` / `if text:` — an empty string is "falsy" in Python, so this is how blank
  or image-only pages get silently dropped.
- `" | ".join(cells)` — join a list of strings with a separator between them (the separator comes
  *first*, `sep.join(list)`, which trips everyone up the first time).

**Bugs hit (this was the buggiest single function in the project, three separate mistakes in
sequence):**

1. `texts.append(shape.has_text_frame.text)` — used the **boolean check** (`has_text_frame`, True/
   False) where the actual text-holding attribute (`text_frame`, no `has_`) was needed. Error:
   `AttributeError: 'bool' object has no attribute 'text'`. Fixed by using `shape.text_frame.text`.
2. The table-reading block was nested *inside* `if shape.has_text_frame:` instead of being a sibling
   `if`. A table shape has no text frame, so `has_text_frame` was `False` for tables and the table
   code never ran — 0 rows of table text were ever extracted, silently. **Indentation is logic in
   Python** — nesting one `if` under another makes it a sub-condition, not an independent check. Fix:
   move `if shape.has_table:` to the same indentation level as `if shape.has_text_frame:`, both
   directly under the `for shape in slide.shapes:` loop.
3. After fixing #2, a leftover `texts.append(shape.text_frame.text)` line was still sitting *inside*
   the table's row loop from an earlier paste. A table is a `GraphicFrame`, which has no
   `text_frame` attribute at all → `AttributeError: 'GraphicFrame' object has no attribute
   'text_frame'`. Fixed by deleting the stray line.

Net lesson from all three: when a shape can be one of several types (text box vs. table vs. image),
each type check should be an **independent sibling `if`**, and every line inside a block should be
re-read to confirm it actually belongs to that block, not copy-pasted from a different one.

### 4.5 `load_manuals` — the public entry point

```python
def load_manuals(source_dir: str) -> List[RawPage]:
    source = Path(source_dir)
    with open(source / "manifest.json") as f:
        entries = [ManifestEntry(**item) for item in json.load(f)]
    all_pages: List[RawPage] = []
    for entry in entries:
        path = source / entry.file
        if not path.exists():
            logger.warning("manifest file not found, skipping: %s", path); continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            pages = _load_pdf_pages(path, entry.product_name)
        elif suffix == ".pptx":
            pages = _load_pptx_pages(path, entry.product_name)
        else:
            logger.warning("unsupported format %s, skipping: %s", suffix, path.name); continue
        all_pages.extend(pages)
    return all_pages
```

`Path` overloads `/` for path joining (`source / "manifest.json"`). `all_pages.extend(pages)` adds
every element of `pages` (as opposed to `.append(pages)`, which would add the *list itself* as one
element). `logging.warning` with `%s` placeholders is the standard-library logging idiom — the
string is only formatted if that log level is actually enabled, unlike an f-string which always
formats.

**Bug hit:** an earlier draft accidentally defined `load_manuals` **twice** in the same file (a
leftover empty stub from an earlier pass, plus the real one). Python doesn't error on a duplicate
`def` — the second definition silently wins. It happened to work here because the real one came
second, but it's a trap: if the order were reversed, calling the function would fail with a
confusing "missing argument" error pointing at the stub. Fixed by deleting the stub. General rule:
never leave two `def`s with the same name in a file.

### 4.6 `chunk_document` — turning pages into chunks

```python
MIN_PAGE_CHARS = 80
CHARS_PER_TOKEN = 4
SECTION_KEYWORDS = { SectionType.WARRANTY: [...], SectionType.WARNING: [...], ... }

def _classify_section(text: str) -> SectionType:
    lowered = text.lower()
    for section_type, keywords in SECTION_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return section_type
    return SectionType.OTHER

def chunk_document(pages: List[RawPage], chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    max_chars = chunk_size * CHARS_PER_TOKEN
    ...
    for page in pages:
        if len(page.text) < MIN_PAGE_CHARS:
            continue
        pieces = [page.text] if len(page.text) <= max_chars else _split_long_text(page.text, max_chars, overlap_chars)
        for index, piece in enumerate(pieces):
            chunks.append(Chunk(
                chunk_id=f"{page.product_name}_p{page.page_number}_c{index}",
                section_type=_classify_section(piece),
                content_hash=hashlib.sha256(piece.encode("utf-8")).hexdigest(),
                ...
            ))
    return chunks
```

**Why "one slide = one chunk" in the common case:** an early check on the real corpus showed the
median slide is ~500 characters and the max is ~1,900 — under or near the 500-token (~2,000-char)
chunk size limit. Splitting on a fixed token window would risk tearing a product name away from its
claim on the same slide. So the chunker only splits when a page genuinely exceeds the limit
(`_split_long_text`, which packs paragraphs up to the limit with a character-overlap tail so no
sentence is cleanly severed) — and on this corpus, that path almost never runs (0 multi-piece pages
out of ~200 chunks in the first test).

**`_classify_section` is keyword matching, not ML classification** — a deliberate, documented
simplification. It tags each chunk with a rough section type (`spec`, `claim`, `usage`, `warning`,
`warranty`, `other`) purely as metadata for debugging/filtering; nothing downstream depends on it
being exactly right.

**`content_hash = sha256(text)`** is the idempotency key. Same text in, same hash out, always — this
is what lets `run_ingestion` (§4.8) skip chunks that are already in the database instead of
re-embedding everything on every run.

**Python syntax:**
- `SECTION_KEYWORDS.items()` — iterate a dict as `(key, value)` pairs.
- `any(kw in lowered for kw in keywords)` — a **generator expression** inside `any()`. Lazy: stops at
  the first `True` without checking the rest of the list. `kw in lowered` is a substring test.
- `f"{page.product_name}_p{page.page_number}_c{index}"` — an f-string; `{}` embeds an expression
  directly into the string.
- `hashlib.sha256(piece.encode("utf-8")).hexdigest()` — hashing needs *bytes*, so `.encode("utf-8")`
  converts the string first; `.hexdigest()` returns the hash as a hex string.

**Bug hit (tuning, not a crash):** the SPEC keyword list originally included bare `"ml"`, which
matched as a substring inside unrelated words like "sm**ml**ooth" or "war**ml**y", pulling over half
of all chunks into the `spec` bucket. Fixed by requiring a leading space (`" ml"`) so it only matches
as a standalone token. A reminder that keyword rules need word-boundary awareness, not just
substring checks.

### 4.7 `embed_chunks` — the embedding step

```python
def embed_chunks(chunks: List[Chunk]) -> List[EmbeddedChunk]:
    if not chunks:
        return []
    vectors = embed_batch([chunk.text for chunk in chunks], task_type="RETRIEVAL_DOCUMENT")
    return [EmbeddedChunk(**chunk.model_dump(), vector=vector) for chunk, vector in zip(chunks, vectors)]
```

**Design decision:** this function stays a **pure transform** (`Chunk → EmbeddedChunk`) with no
database awareness at all. The "skip chunks we already have" logic deliberately lives one level up,
in `run_ingestion`, because it needs to see both the embedder and the vector store — putting it
inside `embed_chunks` would couple two things that should stay independent and separately testable.

**Python syntax:**
- `chunk.model_dump()` — Pydantic method that converts a model instance to a plain dict of its
  fields. `**chunk.model_dump()` unpacks that dict as keyword arguments into `EmbeddedChunk(...)`,
  then `vector=vector` adds the one new field — "copy every field of `chunk`, plus a vector" without
  listing six field names by hand. This is the direct payoff of `EmbeddedChunk` **inheriting** from
  `Chunk` rather than duplicating its fields.
- `zip(chunks, vectors)` — pairs up two lists element-wise: `(chunks[0], vectors[0]), (chunks[1],
  vectors[1]), ...`. The idiomatic way to walk two parallel lists together.

### 4.8 `run_ingestion` — the orchestrator for this slice

```python
def run_ingestion(source_dir: str) -> dict:
    settings = get_settings()
    pages = load_manuals(source_dir)
    chunks = chunk_document(pages, settings.chunk_size, settings.chunk_overlap)
    already = existing_hashes([c.content_hash for c in chunks])
    new_chunks = [c for c in chunks if c.content_hash not in already]
    stored = store_chunks(embed_chunks(new_chunks))
    return {"pages": len(pages), "chunks": len(chunks), "skipped": len(chunks) - len(new_chunks), "stored": stored}
```

This is where settings are read and where the hash-based skip happens — ties `load → chunk →
(skip already-embedded) → embed → store` into one call, and returns a plain dict summary that
becomes the `POST /ingest` HTTP response body directly.

**Python syntax:** `[c for c in chunks if c.content_hash not in already]` is a **list comprehension
with a filter clause**; `already` is a `set`, so `not in` is an O(1) membership check, not a linear
scan.

---

## 5. Database layer — `app/db/vector_store.py`, `app/db/run_logs.py`, `app/db/schema.sql`

### 5.1 `get_connection` — a hand-written context manager

```python
@contextmanager
def get_connection() -> Generator[psycopg2.extensions.connection, None, None]:
    settings = get_settings()
    conn = psycopg2.connect(settings.supabase_db_url)
    register_vector(conn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
```

**Why write this instead of connecting inline everywhere:** this is the *one* place in the codebase
that knows the database URL and handles the commit/rollback/close lifecycle. Every other DB function
does `with get_connection() as conn: ...` and gets correct transaction behaviour for free.

**Python syntax — this is the first hand-written context manager in the project:**
- `@contextmanager` turns a generator function into something usable in a `with` block. Everything
  **before** `yield` runs when the `with` block is entered; the yielded value is what `as conn`
  binds to; everything **after** `yield` runs when the block exits.
- `try / except Exception: ... / finally:` — the `try` block runs the `with`-block body (via
  `yield`); if anything inside raises, `except` catches it, rolls back, and `raise` with no argument
  **re-raises the same exception** (we don't swallow it, just make sure to roll back first);
  `finally` always runs, closing the connection whether or not an error occurred.
- `register_vector(conn)` teaches psycopg2 how to send/receive pgvector's `vector` type — without it,
  inserting a Python list of floats into a `vector` column raises a type error.

### 5.2 `similarity_search` — the actual RAG query

```python
def similarity_search(query_vector, top_k, product_name=None) -> List[ScoredChunk]:
    sql = """select chunk_id, product_name, section_type, page_number, text,
                     1 - (embedding <=> %s::vector) as similarity
              from manual_chunks"""
    params = [query_vector]
    if product_name:
        sql += " where product_name = %s"; params.append(product_name)
    sql += " order by embedding <=> %s::vector limit %s"
    params.extend([query_vector, top_k])
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [ScoredChunk(chunk_id=r[0], ..., similarity=float(r[5])) for r in rows]
```

**This SQL is the heart of the whole RAG system.** `<=>` is pgvector's cosine-**distance** operator
(0 = identical vectors, 2 = opposite). `1 - distance` converts it to cosine **similarity** (1.0 =
identical, 0 = unrelated), which is what the threshold and coverage report are expressed in.
`order by ... limit %s` sorts every row by distance to the query vector and takes the nearest
`top_k` — with ~200–550 rows this is a full sequential scan, and it's still sub-millisecond, which
is why the schema deliberately has **no vector index** (documented in `schema.sql` as a decision,
not an oversight — an HNSW index would only start mattering in the thousands of rows).

**Security note, deliberate:** SQL fragments are concatenated (`sql += ...`), but **values are never
interpolated into the string** — every value goes through the `params` list and gets substituted by
the driver via `%s`. That's the line between safe parameterised queries and SQL injection.

**Python syntax:** `params: list = [query_vector]` then `.append()`/`.extend()` — building the
parameter list in lockstep with the SQL string as it grows.

**Bugs hit here:**
- `psycopg2.OperationalError: could not translate host name "...pooler.supabase.com"` — the very
  first connection attempt used the literal placeholder text from `.env.example` because the real
  Supabase connection string hadn't been pasted in yet.
- `psycopg2.ProgrammingError: invalid dsn: missing "=" ...` — the connection string in `.env` had a
  typo, `ppostgresql://` (double `p`), so psycopg2 couldn't recognise the URL scheme and fell back to
  trying to parse it as `key=value` pairs.
- The database password itself was forgotten (never set consciously — GitHub OAuth was used to log
  into the Supabase *dashboard*, which is a separate credential from the Postgres *database*
  password). Fixed by resetting it from Supabase's Project Settings → Database page, and — because
  the password had briefly appeared in a pasted terminal error in chat — treating it as leaked and
  rotating it again afterward.

### 5.3 `existing_hashes` and `upsert_chunks` — idempotent storage

```python
def existing_hashes(hashes: List[str]) -> Set[str]:
    if not hashes: return set()
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("select content_hash from manual_chunks where content_hash = any(%s)", (hashes,))
        return {row[0] for row in cur.fetchall()}

def upsert_chunks(chunks: List[EmbeddedChunk]) -> int:
    ...
    execute_values(cur, """insert into manual_chunks (...) values %s
                            on conflict (chunk_id) do update set ... = excluded. ...""", rows)
```

`on conflict (chunk_id) do update set ...` is Postgres's **upsert**: re-ingesting a manual overwrites
the existing row (by primary key) instead of erroring on a duplicate key. `execute_values` sends all
rows as one multi-row `INSERT ... VALUES (...), (...), (...)` statement — one network round-trip
instead of one per row.

**Python syntax:** `{row[0] for row in cur.fetchall()}` is a **set comprehension** (curly braces, no
`key: value`), building a `set` for O(1) `in`/`not in` checks. `(hashes,)` is a **one-element
tuple** — the trailing comma is what makes it a tuple rather than just `hashes` in parentheses;
psycopg2 needs it as a tuple to bind as a single query parameter that becomes a Postgres array.

### 5.4 `run_logs.py` and `schema.sql`

`log_run` inserts one row per scoring run into `score_runs`, storing the full scorecard as `jsonb`
plus the raw brief/script text — so any run can be looked up and audited later by its `run_id`.
Called from the orchestrator wrapped in `try/except Exception: logger.exception(...)` — **a
persistence failure must never turn a successful scoring result into a 500**; the user already has
their scorecard, a database hiccup shouldn't take that away.

`schema.sql` holds the two `create table if not exists` statements plus
`create extension if not exists vector;`, run once by hand in the Supabase SQL editor (there's no
migration framework — deliberately, for a project this size).

---

## 6. Retrieval pipeline — `app/pipeline/retrieval.py`

```python
def retrieve_for_claims(claims: List[Claim], top_k: int) -> List[ClaimRetrieval]:
    results = []
    for claim in claims:
        query_vector = embed(claim.text, task_type="RETRIEVAL_QUERY")
        chunks = similarity_search(query_vector, top_k=top_k, product_name=None)
        results.append(ClaimRetrieval(claim_id=claim.claim_id, chunks=chunks))
    return results

def compute_retrieval_confidence(retrievals, threshold) -> List[CoverageEntry]:
    report = []
    for item in retrievals:
        best = max((c.similarity for c in item.chunks), default=0.0)
        status = RetrievalStatus.WELL_GROUNDED if best >= threshold else RetrievalStatus.LOW_CONFIDENCE
        report.append(CoverageEntry(claim_id=item.claim_id, best_similarity=round(best, 4), status=status))
    return report
```

**Why retrieval runs *per claim*, not once for the whole script:** searching with an entire ad
script as the query gives mushy, unfocused results — the vector is an average of everything the
script says. Extracting claims first (§8) and running one targeted query per claim gives each claim
its own precise evidence and its own verdict, with a source chunk id attached for citation.

**`product_name=None` — a known, documented simplification.** `Claim.product_reference` holds free
text from the LLM (e.g. `"Alltimate PDRN Hyalu 7 Serum"`), which does not match the internal product
slug (`alltimate_pdrn_hyalu_7_serum`). Scoping the SQL `where product_name = ...` to that free text
would return zero rows every time — this was actually tried and caused every claim to come back
`top match None (0.000)` in an early test. The line is still in the code, commented out, as a
reminder of what a real fix (resolving free-text names to slugs) would restore.

**`compute_retrieval_confidence` is pure logic — no LLM, no database call.** This is deliberate: it's
what separates "this claim is false" from "we couldn't find the right manual section," and it needs
to be cheap and deterministic so it runs on every single claim of every request with no added cost
or latency.

**Python syntax:**
- `max((c.similarity for c in item.chunks), default=0.0)` — `max` over a generator expression, with
  `default=` so an empty list (a claim whose retrieval genuinely came back with nothing) doesn't
  raise `ValueError: max() arg is an empty sequence`.
- `A if cond else B` — Python's ternary/conditional expression.

**Threshold calibration (a real, data-driven decision, not a guess left unchecked):** with the
initial `SIMILARITY_THRESHOLD=0.6`, a test claim about a completely unrelated product (a waterproof
jacket, nothing like this) scored 0.62 and was marked `well_grounded` — clearly wrong. Real matches
in the corpus scored 0.74–0.81; genuine nonsense scored ~0.58. The threshold was raised to **0.68**,
placing real hits safely above and nonsense/unrelated claims safely below, with margin both ways.
Documented as corpus/model-specific and needing re-calibration if either changes.

---

## 7. LLM client — `app/clients/llm_client.py`

### 7.1 The three-provider saga

The plan called for **GLM via Z.AI** because it was believed to have a free tier. In practice:

1. **Z.AI GLM** → first real call returned `429 ... Insufficient balance or no resource package.
   Please recharge.` It is not free for a fresh account; pay-per-token from the start.
2. **Switched to Gemini's OpenAI-compatible endpoint** (`generativelanguage.googleapis.com/v1beta/
   openai/`, model `gemini-3.6-flash` after `gemini-2.5-flash` turned out to be retired for new
   users — caught via a `404 ... no longer available to new users`). Worked initially, but hit a
   **daily** quota: `429 ... GenerateRequestsPerDayPerProjectPerModel-FreeTier ... limit: 20`. Not a
   per-minute limit that a retry-with-sleep could ride out — a hard 20-requests-per-day ceiling that
   only resets at midnight Pacific. Far too small for a system making 5+ LLM calls per scoring run.
3. **Switched to Groq**, model discovered by actually querying the provider's `/models` endpoint
   (`client.models.list()`) rather than trusting a remembered model name, since Groq had retired the
   Llama 3.3 models that were expected to be there. Settled on `openai/gpt-oss-120b`.

**Why this matters architecturally, not just as trivia:** the client code (`llm_client.py`) **did
not change at all** across all three providers — only three `.env` values did
(`LLM_BASE_URL`/`LLM_API_KEY`/`LLM_MODEL`). This is the payoff of the "thin wrapper, OpenAI SDK
against any compatible base URL" design decided at the very start of the project, proven under real
failure conditions rather than just asserted in a README.

Also renamed the settings fields from `glm_*` to provider-neutral `llm_*` (`llm_api_key`,
`llm_base_url`, `llm_model`) partway through, once it became clear the config was going to hold
whatever provider was active, not specifically GLM — a naming cleanup done consistently across
`config.py`, `.env`, `.env.example`, and `llm_client.py` in one pass.

### 7.2 `generate` and `generate_structured`

```python
def generate(prompt: str, system: str = "You are a precise assistant.") -> str:
    settings = get_settings()
    for attempt in range(MAX_429_RETRIES):
        try:
            response = _client().chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return response.choices[0].message.content or ""
        except RateLimitError:
            if attempt == MAX_429_RETRIES - 1: raise
            time.sleep(RETRY_SLEEP_SECONDS)
    return ""

def generate_structured(prompt: str, schema: Type[T]) -> T:
    system = "... Respond with a single JSON object only ... schema:\n" + json.dumps(schema.model_json_schema())
    for attempt in range(MAX_ATTEMPTS):
        raw = generate(prompt, system=system)
        try:
            return schema.model_validate_json(_strip_fences(raw))
        except (ValidationError, ValueError) as exc:
            last_error = exc
            prompt = f"{prompt}\n\nYour previous reply was invalid: {exc}\nReturn ONLY valid JSON matching the schema."
    raise last_error
```

**Why `generate_structured` is the important one:** every pipeline stage wants *typed* output, not
free text to be parsed by hand. `schema.model_json_schema()` converts a Pydantic model class directly
into the JSON Schema shown to the LLM in the system prompt — so the same class defines both "what we
validate against" and "what we tell the model to produce." Add a field to `StructuredBrief` and the
prompt updates itself automatically; there is no separate schema description to keep in sync.

**The retry-with-feedback loop is deliberate and bounded (2 attempts, not infinite):** if the model's
JSON fails validation (missing field, `"score": "eight"` instead of an int, a verdict string outside
the enum), the exact validation error is fed back into the prompt and it gets one more try. A model
that still can't produce the shape after being told exactly what was wrong is treated as a real
failure, not silently retried forever.

**`_strip_fences`** handles models that wrap JSON in ` ```json ... ``` ` code fences even when told
not to — a common and otherwise-harmless model habit that would break `model_validate_json` if not
stripped first.

**Python syntax:**
- `T = TypeVar("T", bound=BaseModel)` and `def generate_structured(prompt, schema: Type[T]) -> T`
  is a **generic function**: `Type[T]` means "a class, not an instance"; returning `T` means "an
  instance of whichever class was passed in." So `generate_structured(p, StructuredBrief)` is typed
  as returning `StructuredBrief`, and an editor will autocomplete `.target_audience` on the result.
- `except (ValidationError, ValueError) as exc:` — catching **either** exception type via a tuple.
- `response.choices[0].message.content or ""` — `x or ""` yields `""` if `x` is `None`/falsy, a
  guard against a null response body.

### 7.3 The rate-limit-and-token-budget problems, and their fixes

Two separate capacity problems were hit and fixed differently, because they were different kinds of
limit:

**(a) Per-minute 429s** (embedding calls hit Gemini's `embed_content_free_tier_requests` limit of
100/min; later, LLM calls during the eval-dataset build hit similar per-minute throttling). Fixed
with a `try/except RateLimitError: sleep; retry` loop, bounded to `MAX_429_RETRIES = 3`, plus
proactive pacing (`BATCH_PAUSE_SECONDS` between embedding batches) so the limit is rarely even hit.

**(b) A hard per-request token cap, discovered live on the deployed Cloud Run service:**

```
openai.APIStatusError: Error code: 413 - Request too large for model `openai/gpt-oss-120b` ...
tokens per minute (TPM): Limit 8000, Requested 11973
```

This happened in `score_claim_validity` (see §9) once `RETRIEVAL_TOP_K` was raised from 5 to 8 —
more retrieved chunks per claim meant a bigger prompt, and a 7-claim script pushed the prompt past
Groq's free-tier 8,000-token cap. **A `sleep`-and-retry loop cannot fix this kind of error** — the
request itself is too big, retrying it unchanged fails identically every time. The real fix had to
cap the *size of the prompt*, independent of how many claims a script has (see §9's evidence-budget
fix). This is the clearest example in the project of "the size of the input scales with something we
don't control (a script's claim count), so the system has to defend itself with a budget, not a
retry."

---

## 8. Brief parsing and claim extraction

### 8.1 `app/pipeline/brief_parser.py`

```python
def parse_brief(brief_text: str) -> StructuredBrief:
    result = generate_structured(PROMPT.format(brief=brief_text), StructuredBrief)
    return result
```

One LLM call, one job: turn free-form brief text into the explicit fields
(`target_audience`, `key_message`, `mandatory_inclusions`, `tone`, `cta`) that every later scoring
step compares the script against. This is deliberately a **separate, explicit step** rather than
folding brief comparison into a single "score everything" prompt — because having `mandatory_
inclusions` as an actual list lets the alignment scorer report exactly which ones are missing,
which is far more auditable than a vague paragraph of feedback.

The prompt explicitly says *"do not invent requirements it does not state"* — a guardrail against
the model hallucinating a mandatory inclusion the brief never asked for, which would then unfairly
penalise a script for "missing" something that was never actually required.

**Bug hit:** `run_pipeline` in the orchestrator originally called `parse_brief(brief_text)` **twice
in a row** (an accidental duplicate line from an edit), doubling the LLM call and the latency of
every single request for no reason. Found by reading the file, not by a crash — it "worked," just
wastefully. Fixed by deleting the duplicate line.

### 8.2 `app/pipeline/claim_extractor.py`

```python
class _ClaimList(BaseModel):
    claims: List[Claim]

def extract_claims(script_text: str) -> List[Claim]:
    result = generate_structured(PROMPT.format(script=script_text), _ClaimList)
    return result.claims
```

**Why the private `_ClaimList` wrapper exists:** LLMs are far more reliable at producing a JSON
*object* than a bare top-level JSON *array*. So the model is asked for `{"claims": [...]}`, and
`extract_claims` unwraps it before returning — callers of `extract_claims` never see `_ClaimList`,
only the plain `List[Claim]` the rest of the codebase expects. The leading underscore signals "not
meant to be imported elsewhere."

**The prompt's exclusion list grew twice, based on real output, not upfront guessing:**
1. Originally excluded only "marketing flourish" (e.g. "glow like never before"). A real script that
   said "Available now at THE FACE SHOP" had that line extracted as a claim — which can never be
   supported by a product manual (it's a distribution fact, not a product fact), so it was unfairly
   dragging the validity score down. Fixed by adding an explicit exclusion for
   "where the product is sold, its price, or availability."
2. Also told not to paraphrase — "keep the claim close to the script's wording" — because an early
   version of the prompt let the model rephrase claims into its own words, which made it harder to
   trace a claim back to the exact line in the script that made it.

---

## 9. Scoring — `app/pipeline/scoring.py`

Three LLM calls, one per axis, **each prompt explicitly states what NOT to judge** — this is the
mechanism that keeps the three scores independent instead of collapsing into one vague "is this
good" number:

- `BRIEF_ALIGNMENT_PROMPT`: "Judge ONLY brief fit, not writing quality or factual accuracy."
- `MESSAGE_QUALITY_PROMPT`: "Judge ONLY the writing, independent of any brief or of whether claims
  are true."
- `CLAIM_VALIDITY_PROMPT`: "using ONLY the excerpts provided — not your own knowledge" — the
  grounding instruction that makes every `supported`/`contradicted` verdict traceable to an actual
  retrieved chunk id, instead of the LLM answering from its own training data.

### 9.1 `score_claim_validity` and the evidence-budget fix

```python
EVIDENCE_CHUNKS_PER_CLAIM = 4
EVIDENCE_CHAR_BUDGET = 16000
MIN_CHARS_PER_CHUNK = 250

def score_claim_validity(claims, retrievals, coverage) -> ClaimValidityScore:
    if not claims:
        return ClaimValidityScore(score=10, reasoning="... this axis is excluded from the overall score.")
    per_chunk_chars = max(MIN_CHARS_PER_CHUNK, EVIDENCE_CHAR_BUDGET // (len(claims) * EVIDENCE_CHUNKS_PER_CLAIM))
    blocks = []
    for claim in claims:
        chunks = retrieval_by_id[claim.claim_id].chunks[:EVIDENCE_CHUNKS_PER_CLAIM]
        ...
        excerpts = "\n".join(f"...{c.text[:per_chunk_chars]}" for c in chunks)
        blocks.append(...)
    result = generate_structured(CLAIM_VALIDITY_PROMPT.format(claims_block="\n\n".join(blocks)), ClaimValidityScore)
    unverifiable = sum(1 for e in result.claim_verdicts if e.verdict == ClaimVerdict.UNVERIFIABLE)
    if claims and unverifiable / len(claims) > 0.5:
        result.score = min(result.score, 5)
        result.reasoning += " Score capped at 5: most claims could not be checked against the manuals."
    return result
```

**Why this function exists in its current shape — the direct fix for the §7.3(b) 413 error.**
Retrieval fetches `top_k=8` chunks per claim so the coverage report has a full picture, but sending
all 8 full-text chunks for every claim into one prompt scales the prompt size linearly with
`claims × top_k`, with no ceiling — a 7-claim script at top_k=8 produced an ~12,000-token prompt,
over Groq's 8,000 TPM cap. The fix caps **both dimensions independently**:
- only the top 4 chunks per claim (`EVIDENCE_CHUNKS_PER_CLAIM`) go into the prompt, not all 8;
- each chunk's text is truncated to `per_chunk_chars`, computed from a **fixed total character
  budget divided across however many claims and chunks-per-claim there are** — so the prompt size
  stays roughly constant (~4,000 tokens) whether the script has 3 claims or 20, instead of growing
  without bound.

**Python syntax used in the fix:** `//` is integer (floor) division — `16000 // (7 * 4)` gives a
whole number of characters per chunk. `max(MIN, computed)` sets a floor so even a script with many
claims still gets *some* usable evidence per chunk rather than a handful of characters.
`c.text[:per_chunk_chars]` is string **slicing** to a maximum length — safe even when the string is
shorter than the limit.

**Two scoring-policy bugs, both about the same underlying mistake — treating "unmeasurable" as
"perfect":**

1. **No claims → validity score of 10, counted at 40% weight in the overall.** A gibberish/empty
   script (in practice, the literal placeholder text from FastAPI's `/docs` — `"string"` repeated —
   was submitted as a real test) triggered the `if not claims:` early return, which originally
   returned `score=10, reasoning="no checkable claims"`. That 10/10 then got averaged into an
   **overall score of 4.3/10** for pure gibberish — clearly wrong, because "the script says nothing
   checkable" is not the same as "the script is accurate." Fixed at the aggregation level: `claim_
   validity` is **excluded from the weighted average entirely** when there are no claims, and the
   remaining weights (`brief_alignment`, `message_quality`) are renormalised to sum to 1.0 instead
   of quietly leaving the total under 1.0 (see §10).
2. **Empty brief → alignment score of 10/10** with reasoning literally saying *"with no criteria to
   satisfy, the script trivially meets the brief."* Same mistake, mirrored on the other axis: an
   unmeasurable brief was being scored as a perfectly satisfied one. Fixed two ways at once —
   (a) a deterministic check in the orchestrator (§11) that rejects a brief with no usable fields
   before scoring even starts, returning a 422 instead of a fabricated score, and (b) a line added to
   the alignment prompt itself: *"If the brief is empty or states no usable requirements, score 0 ...
   an unmeasurable brief is not a satisfied brief."* Belt-and-braces: the code check catches the
   fully-empty case deterministically and fast (no LLM call needed to detect it), the prompt change
   protects against a partially-empty brief that still passes the code check but is still weak.

**The 50%-unverifiable cap is deliberate business logic living in plain Python, not a prompt
instruction:** "if more than half a script's claims come back `unverifiable`, cap the validity score
at 5" is written as an `if` statement after the LLM call returns, specifically **so it is
predictable and testable** rather than depending on the model reliably following one more
instruction buried in a long prompt. The project's general pattern: anything that must be
deterministic and auditable (scoring caps, weight renormalisation, the empty-brief/no-claims checks)
is plain Python; anything that requires judgement (is this claim true, is this copy persuasive)
is an LLM call.

**Two smaller prompt-quality bugs found from real output and fixed the same way each time (read the
actual scorecard, find the wrong judgement, patch the prompt):**
- Alignment marked a mandatory inclusion "missing" when the script said "salicylic acid" and the
  brief's required phrase was "salicylic acid (BHA)" — an exact-wording match instead of a
  substance match. Fixed by adding: *"Match on substance, not wording... List only inclusions whose
  substance is genuinely absent."*
- The prompt paste that added this fix once landed in the **wrong place** in the string (at the very
  top, splitting the opening sentence in half — `"If the brief is empty...\n, 0-10."` — so the model
  never even saw the "Score how well this script delivers the brief" instruction). Caught by
  re-reading the prompt string after the edit, not by a crash; fixed by moving the sentence to the
  end of the bullet list where it belonged.

---

## 10. `app/pipeline/report.py` — pure aggregation, no LLM

```python
WEIGHTS = {"brief_alignment": 0.3, "message_quality": 0.3, "claim_validity": 0.4}

def build_final_report(alignment, quality, validity, claims, coverage) -> Scorecard:
    axis_scores = {"brief_alignment": alignment.score, "message_quality": quality.score}
    if claims:
        axis_scores["claim_validity"] = validity.score
    total_weight = sum(WEIGHTS[k] for k in axis_scores)
    overall = sum(WEIGHTS[k] * s for k, s in axis_scores.items()) / total_weight
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_") + uuid4().hex[:6]
    return Scorecard(run_id=run_id, overall_score=round(overall, 1), overall_feedback=build_overall_feedback(...), ...)
```

**Why validity gets 0.4 and the other two 0.3 each:** validity is the axis with actual brand/legal
risk (false advertising), so it's weighted highest deliberately — a documented choice, not an
arbitrary default.

**The renormalisation is the fix for the "no claims → 10/10 counted anyway" bug from §9.** Building
`axis_scores` as a dict that only includes `"claim_validity"` **when claims exist**, then dividing
by `sum(WEIGHTS[k] for k in axis_scores)` instead of a hardcoded `1.0`, means: with all three axes
present, `total_weight == 1.0` and the formula is identical to a normal weighted average; with
validity excluded, the remaining two weights (0.3 + 0.3 = 0.6) are scaled back up to sum to 1.0, so
a gibberish script now scores `(0.3×0 + 0.3×1) / 0.6 = 0.5` instead of `4.3`.

**`build_overall_feedback` is deliberately rule-based, not a sixth LLM call.** It assembles a plain-
English summary purely from the structured outputs the other three LLM calls already produced
(missing inclusions, contradicted/unsupported claim ids, a low quality score) — cheaper, faster, and
crucially **cannot hallucinate a problem the scorers didn't actually find**, which a free-form LLM
synthesis call could.

**An accidental regression, caught before it shipped:** at one point, an editor undo (likely a
Cmd+Z reaching back through file history while working in the same file) silently reverted this
whole renormalisation block back to the original fixed-weight formula (`WEIGHTS["claim_validity"] *
validity.score` unconditionally), which would have re-introduced the exact "no claims = 10/10 at
40% weight" bug. It was staged for commit before being noticed. Caught by the habit of running
`git diff --cached` before every commit — this is why that habit is worth keeping even when a change
"should" be trivial.

**Python syntax:**
- `axis_scores.items()` — again, `(key, value)` pairs, here unpacked as `k, s` inside a generator
  expression passed to `sum(...)`.
- `datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_")` — always store/format timestamps in
  UTC, not local time, so run ids and logs are unambiguous regardless of server timezone.
- `uuid4().hex[:6]` — six random hex characters appended to the run id so two runs completing in the
  same second still get distinct ids.

---

## 11. `app/pipeline/orchestrator.py` — `run_pipeline`

```python
def run_pipeline(brief_text: str, script_text: str) -> Scorecard:
    settings = get_settings()
    brief = parse_brief(brief_text)
    if not any([brief.target_audience, brief.key_message, brief.tone, brief.cta]):
        raise ValueError("The brief could not be parsed into any usable fields ...")
    claims = extract_claims(script_text)
    retrievals = retrieve_for_claims(claims, settings.retrieval_top_k)
    coverage = compute_retrieval_confidence(retrievals, settings.similarity_threshold)
    alignment = score_brief_alignment(brief, script_text)
    quality = score_message_quality(script_text)
    validity = score_claim_validity(claims, retrievals, coverage)
    scorecard = build_final_report(alignment, quality, validity, claims, coverage)
    try:
        log_run(scorecard, brief_text, script_text)
    except Exception:
        logger.exception("failed to persist run %s", scorecard.run_id)
    return scorecard
```

This is the single function every interface (the HTTP route, and in principle a future email or MCP
adapter) calls — the whole "brief + script → scorecard" pipeline is ten readable lines because every
step it calls does exactly one job. `any([...])` returns `True` if **any** item in the list is
truthy; since empty strings are falsy, this expression is `False` (triggering the `ValueError`) only
when *all four* structured-brief fields came back empty.

**`try/except Exception: logger.exception(...)` around `log_run`** is the same "a side-effect
failing must not turn a good result into an error" principle as in §5.4 — deliberately broad
`except Exception` here, which is normally a smell, but here it's the intended log-and-continue
policy: persistence failing should never cost the user the scorecard they already have.

---

## 12. Routers — `app/routers/score.py`, `app/routers/ingest.py`

```python
@router.post("", response_model=Scorecard)
def score(request: ScoreRequest) -> Scorecard:
    try:
        return run_pipeline(request.brief, request.script)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
```

Thin adapters over `run_pipeline` — no scoring logic here. `ScoreRequest` (with
`Field(min_length=20)` on both fields) means FastAPI rejects an empty/near-empty body with a 422 and
a clear message before the request even reaches the pipeline. The `except ValueError` here is what
turns the orchestrator's "unusable brief" exception into a proper HTTP error code instead of an
unhandled 500.

`routers/ingest.py` is even thinner: `run_ingestion(MANUALS_DIR)`, one line. **Note the
`data/manuals/` folder does not exist inside the deployed Cloud Run container** (it's excluded via
`.dockerignore`, since the decks are 280 MB of confidential training material) — so `POST /ingest`
is understood to be a **local-only, one-time operation**; the deployed service reads the vector
store that a local ingestion run already populated in Supabase. This is a deliberate architectural
line: ingestion and serving are decoupled through the shared database, not through the filesystem.

---

## 13. Offline retrieval evaluation — `app/pipeline/eval.py`

Two functions answering requirement #5 ("retrieval accuracy eval presented after every run"),
addressed as **two separate things**, because there's no ground truth available at live-request
time:

```python
def build_eval_dataset(sample_size: int, seed: int = 42, pause_seconds: float = 6.5) -> List[EvalQuery]:
    chunks = fetch_all_chunks()
    random.seed(seed)
    sampled = random.sample(chunks, min(sample_size, len(chunks)))
    for chunk in sampled:
        result = generate_structured(QUESTION_PROMPT.format(product_name=chunk.product_name, text=chunk.text), _Question)
        queries.append(EvalQuery(question=result.question, expected_chunk_id=chunk.chunk_id, ...))
    json.dump([...], open(DATASET_PATH, "w"))
    return queries

def run_retrieval_eval(queries: List[EvalQuery], top_k: int = 5) -> OfflineEvalBaseline:
    for query in queries:
        hits = similarity_search(embed(query.question, task_type="RETRIEVAL_QUERY"), top_k=top_k)
        ids = [h.chunk_id for h in hits]
        rank = ids.index(query.expected_chunk_id) + 1 if query.expected_chunk_id in ids else None
        results.append(EvalQueryResult(rank=rank, ...))
    recall = sum(1 for r in results if r.rank is not None) / n
    mrr = sum(1 / r.rank for r in results if r.rank is not None) / n
    ...
```

**Why an LLM generates the ground-truth questions:** for each sampled chunk, an LLM is asked to
write a question whose answer is stated in *that* chunk and would be hard to answer from a
*different* product's manual — turning the chunk id into a known-correct answer, so retrieval
accuracy can be measured without a human writing 40 questions by hand. `random.seed(42)` makes the
sample reproducible across re-runs of the same corpus size.

**Recall@5** = fraction of queries where the expected chunk appears anywhere in the top-5 results.
**MRR** (Mean Reciprocal Rank) = average of `1/rank` (0 if absent) — rewards ranking the right answer
higher, not just finding it somewhere in the top 5.

**Python syntax:** `ids.index(x) + 1 if x in ids else None` — `list.index()` raises `ValueError` if
the item isn't present, so the `in` check guards it first; `+1` converts a 0-based position to a
1-based rank. `sum(1/r.rank for r in results if r.rank is not None)` — a filtered generator
expression as the MRR numerator.

**The corpus grew mid-project (17 → 42 products, 195 → 554 chunks), and the eval was deliberately
re-run rather than reused, because the number means something different at each corpus size:**

| Corpus | Recall@5 | MRR | Cross-product misses |
|---|---|---|---|
| 17 products, 195 chunks | 0.875 | 0.711 | 1 of 5 misses |
| 42 products, 554 chunks | 0.825 | 0.702 | 0 of 7 misses |

The raw recall number went *down* when the corpus tripled — more chunks means more competition for
every query — but the miss analysis (reading each failed query's actual top result, not just the
aggregate number) showed **all 7 misses on the bigger corpus landed on a different slide of the
correct product deck; none went to the wrong product.** For the actual use case (claim verification
needs the right *product's* evidence, not necessarily the one specific expected slide), that's
effectively 40/40 on the metric that matters, despite the headline number looking worse. This is the
project's clearest example of "don't just report the number, read the failures and say what they
actually mean" — and it's explicitly written up that way in the README rather than only quoting
0.825.

One real bug surfaced by this process: **`TFS Brochure.pdf` produced zero chunks** — checked
directly with PyMuPDF and confirmed all 16 pages are scanned images with no text layer at all (not
an ingestion bug, a source-file limitation; OCR would be needed and wasn't in scope). Removed from
`manifest.json` once confirmed, rather than left in silently logging "0 pages" on every future run.

Also surfaced: **duplicate boilerplate slides across decks get stored once**, not once per deck,
because the content-hash-based dedup (§4.6/§4.8) hashes text only, not text+product. A slide that
says only "For internal training use only" appears in many decks with identical text, so the second
and later copies are skipped as "already stored" and keep the product name of whichever deck was
ingested first. Harmless for boilerplate specifically, but a known limitation, documented rather than
silently accepted — the real fix would hash `product_name + text` instead of `text` alone, which
would require clearing and fully re-ingesting the table.

---

## 14. Logging and the frontend

### 14.1 `app/utils/logging.py`

```python
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"time": ..., "severity": record.levelname, "logger": record.name, "message": record.getMessage()}
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)

def configure_logging(level=logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    logging.getLogger("httpx").setLevel(logging.WARNING)
```

**Why JSON lines to stdout, specifically:** Cloud Run ships container stdout straight into Google
Cloud Logging, and if each line is a JSON object, individual fields become filterable/searchable
there (e.g. filter by `run_id`, or by `"severity":"ERROR"`) instead of grepping raw text. The key
name `"severity"` is chosen to match exactly what Cloud Logging looks for to color-code log levels.

**Python syntax:** `class JsonFormatter(logging.Formatter)` with an overridden `format` method is
**subclassing and method overriding** — the standard logging machinery calls `.format(record)` on
whatever formatter is attached to a handler; this class replaces the default plain-text output with
JSON. `record.getMessage()` applies any `%s`-style arguments to the log template — the reason
`logger.info("... %s", x)` was used throughout instead of an f-string: the message is only actually
formatted when needed, and the raw fields stay accessible to any formatter attached later.

### 14.2 `app/static/index.html` — the form

A single static HTML file with vanilla JS (no build step, no framework) served by FastAPI at `/`
via `StaticFiles` + a `FileResponse` route. Two textareas, a "Load example" button, a fetch call to
`POST /score`, and a render function that builds the scorecard view from the JSON response.

**One defensive detail worth noting on its own:** every piece of LLM- or user-supplied text gets
passed through an `esc()` helper before being inserted into the page's HTML:

```js
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
```

Building HTML by string concatenation means any `<` inside a model's reasoning text could otherwise
inject markup into the page. This is the one security-relevant line in the frontend.

**A real bug found via the live deployment, not local testing:** a Safari user hit "Validate script"
and got the generic, unhelpful browser message *"The string did not match the expected pattern"* —
which is Safari's wording for `JSON.parse` failing on a non-JSON response body. The original code
called `res.json()` directly:

```js
const data = await res.json();
if (!res.ok) throw new Error(data.detail || ('HTTP ' + res.status));
```

If the server ever returns something that isn't JSON (an HTML error page during a cold start, a
proxy timeout page), `res.json()` itself throws before the `if (!res.ok)` check is even reached,
producing that opaque browser-native error. Fixed by reading the body as **text first**, then trying
to parse it, with a clear fallback message on failure:

```js
const text = await res.text();
let data;
try { data = JSON.parse(text); }
catch { throw new Error('HTTP ' + res.status + ' - the server returned a non-JSON response. It may be restarting; try again in a few seconds.'); }
if (!res.ok) throw new Error(data.detail || ('HTTP ' + res.status));
```

The actual underlying cause that produced the original error turned out to be a **separate, real
backend bug** — the same 413 token-limit error from §7.3(b)/§9.1 — which is why fixing this frontend
handling alone wasn't the whole story; it made the *real* error visible instead of hiding it behind
a cryptic parse failure, which is exactly what error handling should do.

---

## 15. Deployment — Docker and Cloud Run

### 15.1 `Dockerfile`

```dockerfile
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY data/eval ./data/eval
EXPOSE 8080
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
```

**Why `python:3.12-slim` and not the local dev environment's Python 3.9:** the local `.venv` was
3.9, which is past end-of-life (Google's own libraries print `FutureWarning`s about it on every run
locally) — the container deliberately runs a current, supported version regardless of what's used
for local development.

**Why `requirements.txt` is copied and installed *before* `app/` is copied:** Docker layer caching —
`requirements.txt` changes rarely, application code changes on every commit. Ordering the `COPY`s
this way means a code-only change doesn't force a full dependency reinstall on every build.

**`PYTHONUNBUFFERED=1` matters specifically for Cloud Run:** without it, Python buffers stdout, so
log lines can arrive late or be lost entirely if a container instance is killed before the buffer
flushes.

**`${PORT:-8080}`, `--host 0.0.0.0`:** Cloud Run injects a `PORT` environment variable and expects
the container to listen on it; `--host 0.0.0.0` (not `127.0.0.1`) is required for the container to
be reachable from outside itself at all.

`data/eval/` is deliberately copied into the image (a few KB) — `load_offline_baseline()` reads it
at request time to populate `offline_eval_baseline` in every scorecard, so it has to ship with the
container even though `data/manuals/` (280 MB, confidential) explicitly does not.

### 15.2 `gcloud` setup — a genuinely messy real-world process

This section is worth keeping because it's realistic: cloud CLI setup rarely goes smoothly the first
time, and every failure here had a specific, findable cause.

1. `brew install --cask google-cloud-sdk` — clean install.
2. `gcloud init` was run with a trailing shell comment on the same line
   (`gcloud init          # log in, create or pick a project`) — bash does **not** strip a `#`
   comment that follows a command on the same line the way one might expect, and the comment text
   got passed as literal arguments to `gcloud init`, producing
   `ERROR: (gcloud.init) unrecognized arguments: log in, create or pick a project`. Lesson
   (repeated several times over the course of the project): **never paste an inline explanatory
   comment into a terminal alongside the command it's explaining** — comments belong in files, not
   in a shell prompt.
3. During interactive project creation, `9` was typed both as the **menu choice** ("Create a new
   project") and then, by mistake, again as the **project id itself**, producing
   `project_id must be at least 6 characters long`. Recovered by exiting the wizard (Ctrl+C) and
   running the two steps explicitly instead of through the interactive prompt:
   `gcloud projects create <id>` then `gcloud config set project <id>`.
4. `gcloud billing projects link <project> --billing-account=<id>` initially returned
   `billingEnabled: false` — the link command succeeded, but the billing account itself wasn't
   active (closed/expired), so Cloud Run still couldn't deploy until a working billing account was
   linked. Diagnosed by checking `gcloud billing accounts list`'s `OPEN` column.
5. Multiple `gcloud services enable ...` calls (`run`, `cloudbuild`, `artifactregistry`,
   `secretmanager`) needed to succeed before the first deploy would work at all — each is a
   separate Google Cloud API that has to be turned on per-project.
6. **Several `gcloud run deploy` invocations were mis-typed into the terminal in ways that produced
   confusing, unrelated-looking errors** — worth recording because none of them were code bugs:
   - Running `gcloud run deploy` with **no arguments** puts it into an interactive prompt; pasting
     the *full intended command* as the answer to its "Source code location" question turned the
     whole command line into a single (invalid) folder path, which then got auto-derived into an
     enormous, invalid service name
     (`gpt-oss-120bembeddingprovidergeminigeminiembeddingmodel...`), rejected with
     `Invalid resource name`.
   - A second attempt had **leftover characters from a previous command still on the terminal line**
     before the new command was pasted, concatenating `gcloud run deploygcloud run deploy ...` into
     one invalid token, producing `ERROR: Invalid choice: 'deploygcloud'`.
   - The fix each time was not a code change — it was recognising that gcloud's error was reporting
     exactly what had literally been typed, and re-running the **full explicit command in one clean
     paste** (`gcloud run deploy script-validator --source . --region ... --set-env-vars "..."`)
     rather than answering the interactive wizard piecemeal.
7. **Secrets were created via shell redirection reading straight from `.env`**, not typed by hand,
   specifically to avoid keys passing through the terminal history or being visible in a pasted
   screenshot:
   ```bash
   grep '^LLM_API_KEY=' .env | cut -d= -f2- | tr -d '\n' | gcloud secrets create llm-api-key --data-file=-
   ```
   `cut -d= -f2-` (not `-f2`) matters here because the Supabase connection string and API keys can
   themselves contain `=` characters; `-f2-` takes "everything from the second field onward," not
   just the second field alone. `tr -d '\n'` strips the trailing newline `grep` leaves, since a
   stray newline appended to an API key is a classic cause of "invalid API key" errors that look
   like the key itself is wrong when it isn't.
   Then IAM access was granted per secret:
   ```bash
   gcloud secrets add-iam-policy-binding llm-api-key --member="serviceAccount:<project-number>-compute@developer.gserviceaccount.com" --role="roles/secretmanager.secretAccessor"
   ```
8. **A subtle "it looks deployed but isn't the version you think" trap:** `gcloud run deploy
   --source .` uploads and builds **whatever is currently on local disk at the moment the command
   runs**, completely independent of git and GitHub. It's easy to think "I pushed to GitHub, so the
   live site is updated" — it isn't; GitHub was never wired to trigger anything. Concretely, this bit
   during the project: the evidence-budget fix (§9.1) was written and tested locally, but the live
   Cloud Run revision at the time still had the old, unfixed code and the old `RETRIEVAL_TOP_K=5` env
   var — so a live test correctly reproduced the 413 error that had *already been fixed locally*,
   because the fix had never actually been redeployed yet. The general lesson: **after any code fix
   meant to affect the live service, redeploy and re-test the live URL specifically — a passing local
   test proves nothing about what's currently running in the cloud.**

---

## 16. Git hygiene lessons from this project

A few things worth remembering that had nothing to do with the application code itself:

- **Author identity**: `git config user.name`/`user.email` were never set, so the first six commits
  were authored as `USER <user@RENTKARs-MacBook-Air.local>`. Since these commits hadn't been pushed
  yet, they were safely rewritten with `git filter-branch --env-filter ...` before the first push —
  this is **only** safe pre-push; rewriting history that others have already pulled requires a
  force-push and coordination.
- **`git diff --cached` before every commit** — the habit that caught the accidental
  renormalisation-revert in `report.py` (§10). A ten-second check before `git commit` is far cheaper
  than shipping a silent regression.
- **`.gitignore`/`.dockerignore` split responsibilities**: `.gitignore` keeps 280 MB of confidential
  manuals and `.env` out of version control entirely; `.dockerignore` (derived from `.gitignore` by
  default, but explicit here) keeps the same things out of the container image even if they were
  ever accidentally tracked. `data/manuals/*` with `!data/manuals/manifest.json` (a leading `!`
  **un-ignores** one specific path inside an otherwise-ignored folder) keeps the curated file list
  in git while the actual decks stay local-only.
- **Secrets that briefly appear in a pasted terminal error or chat message should be treated as
  leaked and rotated**, even if the exposure looks minor — this happened once with the Supabase
  database password and it was reset immediately after being noticed, rather than assumed safe
  because "it was only shown once."

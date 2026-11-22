# Product Flow — Creative Script Validation Service

A high-level walk-through of what happens, step by step, from a product manual landing in a Drive
folder to a creator getting a scorecard back. For architecture rationale and line-by-line detail see
`DEVELOPMENT_JOURNAL.md` and `README.md`; this doc is the flow only.

There are two independent flows: **Ingestion** (run once, offline, whenever manuals change) and
**Scoring** (run per request, whenever a brief + script comes in). They only share one thing: the
Supabase database sitting between them.

---

## Flow A — Ingestion (offline, one-time / occasional)

```
Drive folder (product manuals)
        │  manually downloaded
        ▼
data/manuals/*.pptx, *.pdf  +  manifest.json (curated list + clean product names)
        │
        ▼
[1] load_manuals()         → reads manifest, opens each file, extracts raw text per page/slide
        │
        ▼
[2] chunk_document()       → one slide = one chunk (splits only if a page is unusually long),
                              tags each chunk with a rough section type (spec/claim/usage/etc.)
        │
        ▼
[3] embed_chunks()         → sends chunk text to Gemini, gets back a 768-number vector per chunk
        │
        ▼
[4] store_chunks()         → upserts into Supabase (Postgres + pgvector), skipping chunks
                              already stored (so re-running ingestion after adding new files
                              only processes the new ones)
        │
        ▼
Supabase: manual_chunks table (554 rows / 42 products currently)
```

**Trigger:** `POST /ingest` (local only — this is not exposed usefully on the deployed Cloud Run
service, since the manual files themselves aren't shipped in the container; only the resulting
database rows are used at serving time).

**Also produced from this same data, separately, as a quality check:**

```
Supabase manual_chunks
        │
        ▼
[5] build_eval_dataset()   → LLM writes one question per sampled chunk, so the "correct answer"
                              (that chunk's id) is known in advance
        │
        ▼
[6] run_retrieval_eval()   → runs each question through the same retrieval used at scoring time,
                              checks whether the correct chunk came back, computes two numbers:
                              Recall@5 and MRR
        │
        ▼
data/eval/offline_eval_report.json   (current: Recall@5 = 0.825, MRR = 0.702, 40 questions)
```

This is the offline half of the "retrieval accuracy" requirement — a standing baseline for the
retrieval system in general, checked whenever the corpus or chunking changes. It's included in
every scorecard's `offline_eval_baseline` field automatically.

---

## Flow B — Scoring (live, per request)

```
User opens the web form (or calls POST /score directly)
Pastes a campaign brief + a script → clicks "Validate"
        │
        ▼
[1] parse_brief()               → LLM converts the free-text brief into structured fields:
                                    target audience, key message, must-include items, tone, CTA
        │   (if the brief has no usable content at all → request rejected here, HTTP 422)
        ▼
[2] extract_claims()            → LLM pulls out every factual, checkable statement from the
                                    script (ingredients, %, certifications, results — not
                                    marketing flourish, not price/availability)
        │
        ▼
[3] retrieve_for_claims()       → for EACH claim separately: embed it, search Supabase for the
                                    most similar manual chunks (top 8)
        │
        ▼
[4] compute_retrieval_confidence()  → for each claim, is the best match similarity above 0.68?
                                        flags each claim well_grounded or low_confidence
        │
        ├──────────────┬──────────────────────┬─────────────────────────┐
        ▼              ▼                      ▼                         │
[5] score_brief_    [6] score_message_    [7] score_claim_validity()    │
    alignment()          quality()             (uses the retrieved      │
    "does this          "is this good           manual excerpts from    │
    deliver the          advertising,            step 3, gives each     │
    brief?"               on its own merits,      claim a verdict:      │
                          regardless of the        supported /          │
                          brief or the facts?"     unsupported /        │
                                                    contradicted /       │
                                                    unverifiable)        │
        │              │                      │                         │
        └──────────────┴──────────────────────┴─────────────────────────┘
                              │
                              ▼
        [8] build_final_report()   → combines all three scores into one weighted overall
                                       score (validity weighted highest, 40%, since it's the
                                       axis with real brand/legal risk), writes a plain-English
                                       overall_feedback summary, attaches the retrieval
                                       confidence report and the offline eval baseline
                              │
                              ▼
        [9] log_run()   → saves the full scorecard + original inputs to Supabase (score_runs
                           table), so any run can be looked up later by its run_id
                              │
                              ▼
                    Scorecard JSON returned to the browser
                    → rendered as scores + reasoning + a claim-by-claim evidence table
```

**Three scores, one feedback line, per request:**

| Axis | Question it answers | Weight |
|---|---|---|
| Brief alignment | Does the script do what the brief asked? | 30% |
| Marketing message quality | Is it good advertising, on its own merits? | 30% |
| Product claim validity | Are the factual claims actually true, per the manuals? | 40% |

If a script makes no checkable claims, or a brief has no usable requirements, that axis is excluded
from the overall average rather than silently scored as perfect — an unmeasurable axis is not a
passing axis.

**Cost per request:** 5 LLM calls (steps 1, 2, 5, 6, 7) + one embedding call per claim (step 3).
Takes roughly 20–30 seconds end to end.

---

## Where things live

| Component | What it is | Where it runs |
|---|---|---|
| Web form | Two textareas + results view | Served by the same backend, at `/` |
| API (`/score`, `/ingest`, `/health`) | FastAPI app | Google Cloud Run |
| Vector store + run history | Postgres + pgvector | Supabase (external, shared by local + cloud) |
| LLM calls | Chat completions (brief parsing, claim extraction, scoring) | Groq (swappable — see below) |
| Embeddings | Text → vector | Gemini |

**Provider-swap note for planning purposes:** the LLM has been run against three different providers
during development (a paid one that turned out not to be free, then Gemini, then Groq) with **zero
code changes** each time — only three configuration values change. This means the system isn't
locked into any one LLM vendor; a future move to a paid tier, a different model, or an in-house model
behind an OpenAI-compatible endpoint is a config change, not a re-build.

---

## Known gaps (for planning, not hidden)

- **Claims aren't scoped to a specific product before retrieval.** The claim extractor identifies a
  product name from the script as free text, which doesn't yet match the internal product catalogue
  ids, so retrieval currently searches the whole catalogue for every claim rather than just the
  named product. It still finds the right product's evidence almost every time in testing, but
  scoping would make it stricter.
- **Ingestion has to be run manually, locally**, whenever new manuals are added — there's no
  automatic sync from the Drive folder.
- **Only one interface exists today** (the web form / direct API call). The architecture supports
  adding email or a chat-agent interface (MCP) as thin wrappers over the same core scoring function
  without touching the scoring logic itself, but those wrappers haven't been built yet.

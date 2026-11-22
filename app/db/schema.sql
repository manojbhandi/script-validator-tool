-- Run once in the Supabase SQL editor.

create extension if not exists vector;

-- Ingested manual chunks. embedding dimension must match EMBEDDING_DIMENSIONS.
-- No vector index: the corpus is ~200 rows, so a sequential scan is already sub-millisecond.
-- Add an HNSW index here if the corpus grows into the thousands.
create table if not exists manual_chunks (
    chunk_id      text primary key,
    product_name  text not null,
    section_type  text not null,
    page_number   int  not null,
    text          text not null,
    content_hash  text not null,
    embedding     vector(768) not null,
    created_at    timestamptz not null default now()
);

create index if not exists manual_chunks_product_idx on manual_chunks (product_name);
create index if not exists manual_chunks_hash_idx on manual_chunks (content_hash);

-- One row per scoring run, for auditing and trend tracking.
create table if not exists score_runs (
    run_id         text primary key,
    created_at     timestamptz not null default now(),
    overall_score  numeric(4,1) not null,
    brief_text     text not null,
    script_text    text not null,
    scorecard      jsonb not null
);

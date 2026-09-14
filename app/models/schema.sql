create table if not exists score_runs (
    run_id         text primary key,
    created_at     timestamptz not null default now(),
    overall_score  numeric(4,1) not null,
    brief_text     text not null,
    script_text    text not null,
    scorecard      jsonb not null
);


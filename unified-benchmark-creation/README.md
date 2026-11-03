# Unified benchmark creation

This directory constructs the multi-product shopping requests used by RecoAtlas. It is separate
from `unified-tool-creation/`: that package creates atomic product queries and trains recommendation
tools, while this package mines behavioral reference sets and writes one natural shopping request
for each retained set.

The implementation covers the four paper environments with one pipeline:

- Amazon Musical Instruments
- Amazon Electronics, using the already filtered 2020-or-later catalog
- Amazon Video Games
- H&M Clothing

## What the pipeline does

1. Assign every user to the protected, deterministic `shared_train`, `agent_dev`, or
   `agent_hidden` partition.
2. Collapse repeated user–item interactions to the latest event.
3. Select the first largest inclusive seven-day window for each user, retaining its seven most
   recent distinct products when necessary.
4. Require 3–7 products and at least two known subcategories. H&M additionally requires distinct
   product codes, distinct product types, and one merchandising group; Ladieswear and Divided are
   treated as one group.
5. Present each candidate to the coherence judge three times in independently randomized order.
   A vote passes only for `PASS`, non-low confidence, and no conflicting item. A candidate needs
   two passing votes.
6. Generate a 12–55 word request, apply deterministic leakage checks, and run a separate
   query–set consistency judgment. Up to eight attempts are allowed.
7. Remove exact normalized-query and identical-reference-set overlap, then write the clean splits.

Neither retrieval tools nor recommendation-agent outputs are consulted during construction.

## Install

```bash
cd unified-benchmark-creation
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
```

Set `OPENAI_API_KEY` and, for a compatible non-OpenAI gateway, `OPENAI_BASE_URL`. Set
`RECOATLAS_BENCHMARK_MODEL` to the deployment name. The paper used the managed alias
`gpt-5.6-luna`; an alias is not a pinned downloadable checkpoint, so save the returned model and
system fingerprint recorded in `calls/`.

## Protected split

Use the exact same private split manifest for benchmark creation and tool training. It is produced
once by:

```bash
python ../unified-tool-creation/scripts/create_benchmark_split.py \
  --domain amazon --category Musical_Instruments \
  --interactions /path/to/interactions.parquet \
  --output private_splits/musical-instruments.json
```

The manifest contains user IDs and the secret split key. Never commit or publish it. Recreating a
manifest without the original key produces a different valid benchmark, not the released split.

## Amazon input

Run stage 1 of `unified-tool-creation` first. Pass its `data/` directory, containing
`interactions.parquet`, `product_texts.json`, and `subcategories.json`:

```bash
recoatlas-create-benchmark all \
  --profile amazon-musical-instruments \
  --split-manifest private_splits/musical-instruments.json \
  --amazon-data-dir /large-disk/tool-run/data \
  --run-dir /large-disk/benchmark-runs/amazon-musical-instruments-v1
```

The Electronics and Video Games profiles use the same command with their corresponding canonical
tool data and private split. The Electronics catalog must already have the paper's 2020-or-later
item-availability filter; benchmark creation does not reinterpret review timestamps as catalog
availability.

## H&M input

Pass the public competition's article and transaction tables directly. Article IDs remain
zero-padded strings.

```bash
recoatlas-create-benchmark all \
  --profile hm-clothing \
  --split-manifest private_splits/hm-clothing.json \
  --hm-articles /data/articles.csv \
  --hm-transactions /data/transactions_train.csv \
  --run-dir /large-disk/benchmark-runs/hm-clothing-v1
```

For easier recovery, run `mine`, `judge`, `generate`, and `export` separately with the same
arguments. Successful API calls are append-only and keyed by candidate, vote, and attempt; reruns
reuse them. A changed profile, model, split fingerprint, or frozen prompt is rejected in an
existing run directory.

## Outputs

The local run retains the audit trail:

```text
RUN/
├── candidates.jsonl
├── coherence_decisions.jsonl
├── query_decisions.jsonl
├── calls/
├── generation_policy.json
├── summary.json
└── release/
    ├── shared_train.jsonl
    ├── agent_dev.jsonl
    └── agent_hidden.jsonl
```

Only `release/*.jsonl` is public. Each line contains exactly:

```json
{"candidate_id":"...","partition":"agent_hidden","query":"...","positive_items":["...","...","..."]}
```

The public files contain no user IDs, protected split keys, raw interactions, prompts, LLM call
logs, generation metadata, manifests, or auxiliary metadata files.

## Validate and publish

Authenticate with Hugging Face or set `HF_TOKEN`, then inspect the operation:

```bash
recoatlas-publish-benchmark \
  --profile hm-clothing \
  --release-dir /large-disk/benchmark-runs/hm-clothing-v1/release \
  --dry-run
```

Remove `--dry-run` to publish to `iaouali/recoatlas`. The publisher replaces only the selected
environment's `data/` prefix and removes stale files within that prefix. It cannot modify the
environment's `tools/` or `benchmark/` directories.

## Tests

```bash
pytest -q
```

The tests use synthetic data and make no network or model calls.

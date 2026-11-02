# Unified tool creation

This directory rebuilds the recommendation tools consumed by RecoAtlas. It
provides one interface for the four paper environments:

- Amazon Musical Instruments
- Amazon Video Games
- Amazon Electronics, restricted to products first available in 2020 or later
- H&M Clothing

The training code uses only the `shared_train` user partition. It produces a
catalog, frozen Qwen3 item embeddings, a query-encoder LoRA retrieval model,
and complementarity embeddings. The clean publisher validates and uploads
exactly the five files expected by the RecoAtlas environment.

## 1. Install

Python 3.11 or newer and a CUDA-capable machine are required for a full build.
From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Large datasets, model caches, and checkpoints are written below
`RECO_SCRATCH_ROOT` (default: `/tmp/reco-atlas`). Set it to a location with at
least 55 GiB available for one build:

```bash
export RECO_SCRATCH_ROOT=/path/to/large/scratch/reco-atlas
```

## 2. Create the protected user split

The tools must use the same deterministic, user-disjoint 70/15/15 split as the
benchmark. The generated manifest contains user identifiers and the private
split key: keep it local and never upload it.

Amazon input must expose `user_id` and `rating` columns:

```bash
python scripts/create_benchmark_split.py \
  --domain amazon \
  --category Musical_Instruments \
  --interactions /path/to/amazon/interactions.parquet \
  --output private_splits/musical-instruments.json
```

For H&M, point `--interactions` to `transactions_train.csv` (or a directory of
CSV/Parquet transaction shards):

```bash
python scripts/create_benchmark_split.py \
  --domain hm \
  --category H_and_M \
  --interactions /path/to/transactions_train.csv \
  --output private_splits/hm-clothing.json
```

If reproducing an existing release, pass its original private key with
`--salt`; a newly generated key creates a different, internally consistent
partition.

## 3. Configure catalog-query generation

Stage 2 converts eligible catalog items into atomic retrieval queries. Either
provide an existing `product_queries.json` with
`--query-seed /path/to/product_queries.json --query-cache-only`, or configure an
OpenAI-compatible endpoint:

```bash
cp .env.example .env
# Set OPENAI_API_KEY, optionally OPENAI_BASE_URL, and RECOATLAS_QUERY_MODEL.
```

Catalog text is treated as untrusted data by the query-generation prompt.

## 4. Train a profile

Inspect a frozen profile:

```bash
python scripts/train_tools.py amazon-musical-instruments --show-profile
```

Run all stages for an Amazon profile:

```bash
python scripts/train_tools.py amazon-musical-instruments \
  --split-manifest private_splits/musical-instruments.json \
  --run-id musical-instruments-v1
```

The other Amazon profiles are `amazon-video-games` and
`amazon-electronics`. Amazon data are read from the pinned
`McAuley-Lab/Amazon-Reviews-2023` revision in the profile.

For H&M, download the public competition's `articles.csv` and
`transactions_train.csv`, then run:

```bash
python scripts/train_tools.py hm-clothing \
  --articles /path/to/articles.csv \
  --transactions /path/to/transactions_train.csv \
  --split-manifest private_splits/hm-clothing.json \
  --run-id hm-clothing-v1
```

The completed run is stored under
`$RECO_SCRATCH_ROOT/runs/<category>/<run-id>/`. Use `--dry-run` to inspect a
launcher without starting training. The frozen numerical settings are defined
in `scripts/train_amazon_tools.py` and `scripts/train_hm_tools.py`.

## 5. Validate and publish the tools

Set `HF_TOKEN` or authenticate with `huggingface-cli login`. First validate
locally without network access:

```bash
python scripts/publish_tools.py amazon-musical-instruments \
  --run-dir "$RECO_SCRATCH_ROOT/runs/Musical_Instruments/musical-instruments-v1" \
  --dry-run
```

Then publish:

```bash
python scripts/publish_tools.py amazon-musical-instruments \
  --run-dir "$RECO_SCRATCH_ROOT/runs/Musical_Instruments/musical-instruments-v1"
```

By default the profiles target the `iaouali/recoatlas` Hugging Face dataset.
Use `--repo namespace/repository` to target another dataset repository. The
publisher synchronizes only the selected environment's `tools/` prefix; it
cannot modify sibling `data/` or `benchmark/` folders.

Each environment receives exactly this serving layout:

```text
<environment>/tools/
├── catalog/
│   ├── product_texts.json
│   └── subcategories.json
├── embeddings/
│   ├── item_embeddings.pt
│   └── complementarity_embeddings.pt
└── retrieval/
    └── retrieval_model.pt
```

No raw interactions, protected split identifiers, mining caches, training
recipes, intermediate checkpoints, or experiment logs are published.

## Tests

```bash
pytest -q tests
```

# Benchmark artifacts

The repository contains code and pinned download metadata, not dataset copies
or trained tensors. `prepare_artifacts.py` downloads a category into:

```text
artifacts/<category>/
  data/product_texts.json
  data/subcategories.json
  embeddings/item_embeddings_qwen.pt
  embeddings/comp_embeddings_qwen.pt
  models/recsys_model_qwen.pt
  queries/<published-partition>.jsonl
```

The source repositories and 40-character revisions are in
`artifact_sources.json`. Large files have SHA-256 checksums; regular Git files
have Git blob SHA-1 identifiers (hashing the Git blob header plus file content).
All downloads and runtime inputs are checked. Modified files are rejected,
not overwritten. Catalog order must match every embedding array.

| Category | Catalog items | shared_train | agent_dev | agent_hidden |
| --- | ---: | ---: | ---: | ---: |
| amazon-musical-instruments | 62,961 | 8,937 | 2,028 | 1,960 |
| amazon-electronics | 89,371 | — | — | 339 |
| amazon-video-games | 42,818 | — | — | — |
| hm-clothing | 104,993 | 9,019 | 2,028 | 1,960 |

These are counts of the pinned publications, not inferred counts. The published
Musical Instruments shared_train file has 8,937 rows, not the 9,019 cited in the
appendix; this does not change its 1,960-query hidden evaluation. Unpublished
partitions are not generated from predictions or silently borrowed elsewhere.
Reference IDs for the paper's 771 Video Games queries remain unavailable in
these publications. `--tools-only` supports downloading its tools, but the
benchmark runner refuses to claim exact-recovery evaluation without its queries.

Query records contain `candidate_id`, `partition`, `query`, and
`positive_items`. Additional construction metadata may also be present. The
runner supplies only the query string and `len(positive_items)` to the agent.
Gold IDs are used by the evaluator after reporting, not by search, expansion,
near-duplicate pruning, or the LLM judge. H&M article IDs remain ten-digit,
zero-padded strings.

Some catalog products lack a known subcategory. The paper tools retain their
recorded handling of this missing metadata; the verifier does not invent labels
or discard these catalog items.

## Models

The retrieval checkpoint is a custom PyTorch dictionary with `model_config`
and `adapter_state_dict`, not a standalone Transformers/PEFT model directory.
The loader instantiates the pretrained query encoder, installs the matching
LoRA layers, and loads their selected weights. Cached item vectors remain fixed.
Semantic retrieval disables those same adapter layers.

The query encoder is `Qwen/Qwen3-Embedding-0.6B`, pinned to
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, the cached revision recorded during
the supplementary release validation. Original training/loading did not pin
every upstream revision, so this is not a claim of complete training provenance.

Complementarity needs only aligned, frozen anchor/complement embeddings for the
existing catalog. H&M's released arrays correspond to the selected residual
projection checkpoint; its projection-head weights are not included in the
published tool package. That is sufficient for inference on this fixed catalog,
but not for projecting new items or retraining the tool. The release does not
substitute an older MLP checkpoint for the residual model.

The agent weights and hosted judge are separate dependencies. They are not
included in any of these catalog-tool packages.

## Existing traces only

To evaluate or judge saved predictions without downloading tensor files:

```bash
python prepare_artifacts.py --category hm-clothing --metadata-only
python scripts/verify_artifacts.py --category hm-clothing --metadata-only
```

This fetches the catalog, subcategories, and published queries. Running the
agent later requires the full artifact command. Trace files must follow this
runner's JSONL schema; arbitrary legacy experiment dumps are not automatically
converted.

Downloading or using external datasets and models remains subject to their
upstream licenses and access terms. This repository does not relicense them.

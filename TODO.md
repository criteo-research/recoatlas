# Release TODO

## Benchmark-construction parity (blocking for an exact-reproduction claim)

- Run `unified-benchmark-creation` on a machine with large scratch using the original protected
  split manifests and the exact canonical source artifacts. Do not generate new split keys.
- Before making any LLM calls, compare the new mined candidates with the retained paper inputs:
  `candidate_id`, partition, ordered item IDs, and bundle size must match exactly. Check both the
  complete Amazon Musical Instruments candidates and H&M candidates. For H&M, separately verify
  the product-code, product-type, merchandising-group, Ladieswear/Divided, day-tie, and identical-set
  rules.
- Compare SHA-256 hashes of the four frozen strings (coherence prompt, query system prompt,
  consistency prompt, and generation guidance) with the audited paper scripts. A textual mismatch
  requires a new versioned profile rather than silently changing an existing run.
- Reproduce the public split counts and size histograms from a full run:

  | Environment | `shared_train` | `agent_dev` | `agent_hidden` |
  |---|---:|---:|---:|
  | Amazon Musical Instruments | 8,937 | 2,028 | 1,960 |
  | H&M Clothing | 9,019 | 2,028 | 1,960 |

  Also confirm zero exact-normalized-query overlap and zero identical-reference-set overlap across
  the three exported partitions. When Electronics and Video Games are published through the unified
  layout, confirm their 339 and 771 held-out rows respectively.
- Audit several candidates by hand from raw interaction to exported request. Confirm that no user ID,
  split key, retrieval result, agent result, recovery score, or evaluator feedback enters an LLM
  request or a public row.

## Live API and resume validation

- On `agent_dev`, run at least one candidate of every size K=3--7 against the intended managed
  deployment. Verify three randomized coherence calls, the 2-of-3 vote rule, deterministic query
  gates, consistency validation, `finish_reason=stop`, and the recorded response model/system
  fingerprint. This is a protocol smoke test, not evidence that a managed alias is revision-pinned.
- Interrupt a smoke run during coherence calls and again during query generation, then resume it.
  Confirm that successful keyed calls are not repeated, a torn final JSONL append is removed, and
  the resumed public files are byte-identical to an uninterrupted run using the same saved calls.
- Exercise retry exhaustion with a mock endpoint. The run must remain incomplete and publish
  nothing; technical failure must never be treated as semantic rejection.

## Hugging Face publication audit

- First publish each environment to a private staging dataset. Confirm that the commit modifies only
  `<environment>/data/` and that the directory contains exactly the supported split JSONL files—no
  manifests, metadata, raw interactions, caches, prompts, call logs, or protected split files.
- Run a schema/privacy scan on every public row. The only fields must be `candidate_id`, `partition`,
  `query`, and `positive_items`; item IDs must be strings; every set must contain 3--7 unique catalog
  IDs; no candidate ID may repeat.
- After the public HF commit, pin its immutable revision in `src/recoatlas/assets.py`, update every
  file size/checksum and expected row count, and run the environment's real-file load plus one episode
  on each published split. Never point the environment at a mutable HF branch.
- Run `recoatlas-publish-benchmark --dry-run` immediately before the real commit and review every
  add/delete path. Preserve the prior HF revision so an erroneous publication can be recovered.

## Clean-room reproduction

- From a fresh clone with empty caches, create the protected split once and share that exact manifest
  between `unified-tool-creation` and `unified-benchmark-creation`. Verify that tool training consumes
  only `shared_train` while benchmark construction creates only the partitions declared by its
  profile.
- Build Musical Instruments and H&M end to end on large scratch without importing old run folders.
  Record source revisions, the private split fingerprint, code commit, model/deployment name,
  returned model fingerprints, output checksums, and elapsed/call counts in the local audit record.
- Run the environment tests, benchmark-creation tests, tool-creation tests, and one real artifact
  smoke test from that clone. Create a versioned release tag only after all four suites pass.

## Optional grammar validation

- Validate the optional `grammar` extra and its grammar-dependent tests on a machine with enough
  scratch space. The current dependency lock resolves a full Torch/CUDA stack, so both the uv cache
  and the temporary environment must live on a large filesystem. Verify at minimum:

  ```bash
  UV_CACHE_DIR=/large-scratch/recoatlas-uv-cache \
  UV_PROJECT_ENVIRONMENT=/large-scratch/recoatlas-venv \
    uv sync --extra grammar --frozen

  UV_CACHE_DIR=/large-scratch/recoatlas-uv-cache \
  UV_PROJECT_ENVIRONMENT=/large-scratch/recoatlas-venv \
    uv run --frozen pytest -q tests/test_backend.py
  ```

  This remains unvalidated for the release; the non-grammar tests pass.

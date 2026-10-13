# AGENTS.md

Notes for coding agents working on this repo. Read [ELI5.md](ELI5.md) (what it is, how to use it)
and [docs/overview.md](docs/overview.md) (internals, design, measurements) first. This file covers
what those don't say explicitly: invariants, traps, and decisions already tried and rejected.

## Commands

```bash
uv sync --all-extras                 # dev environment
uv run ruff check . && uv run ruff format .
uv run pytest -q                     # fast tests, no downloads
uv run pytest -q -m slow             # real H&M files (~2.8 GB, cached by huggingface_hub)
RECOATLAS_DEVICE=cuda uv run recoatlas-openenv --port 8000
```

- **Build backend:** `uv_build`, src layout.
- **`uv.lock` is committed.** Update it with `uv lock`.

## Invariants

1. **Tool outputs must stay byte-identical to recoatlas-v1.**
   - Same inputs give the same JSON strings, for both tool variants (`utility`, `semantic`).
   - This is the benchmark's definition: any change to scores, rounding, ordering or text
     truncation changes results.
   - Check it on probe calls (search → complementary → near-duplicate) against outputs saved before
     your change, on the same device. The last check was 300/300 identical per variant.
2. **`src/recoatlas/tools/` is copied from recoatlas-v1. Keep changes minimal and listed.**
   - The changes so far: imports and paths, `dtype=torch.float32` in `query_encoder.py`, and
     `search_many` in `trained_retrieval.py`.
   - These files and `baselines.py` are excluded from ruff on purpose. Don't reformat them.
   - When you change one, add it to the list in `docs/overview.md` ("Tool code").
3. **The query encoder must load in float32.** transformers 5 defaults to the checkpoint's bfloat16,
   which changed about two thirds of search results. Don't remove the `dtype=torch.float32`.
4. **Scoring follows v1 and the paper exactly:**
   - first run `reports.prepare_report`: filter duplicate/out-of-catalog IDs and, when the
     candidate list is nonempty, unobserved IDs; backfill short reports in discovery order;
   - malformed schemas fail before backfilling;
   - `sethit` cuts the list to the first K positions *before* dropping duplicates or unknown IDs;
   - the metric receives the processed report, not the raw model selection;
   - `reward_direct` = `sethit` if the processed report is valid, else −1;
   - "seen" = IDs returned by `search_products` or `get_complementary_products` only.
     `get_substitute_products` never adds to seen.
5. **Submit feedback never uses the reference items.** It only checks count, duplicates, catalog
   membership and seen.
6. **The report format is strict:** `product_id` and `reasoning`. Don't accept aliases like
   `item_id`. Preserve the original parser's fallback for absent `reasoning` and its empty-candidate
   edge case; neither should be silently rewritten in a parity refactor.
7. **One `Env` per process, shared by every episode.**
   - Tool methods on `Env` must stay stateless and thread-safe.
   - Per-episode state belongs on `Episode`, or on the caller (Miles keeps its own seen list).

## Where things are defined once

- **Tool descriptions shown to agents:** the `Env` method docstrings plus `SUBMIT_REPORT_DOC`,
  collected in `env.TOOL_DOCS`.
  - `Episode`, the OpenEnv tools and the TRL methods get their `__doc__` assigned from it.
  - TRL builds tool schemas from those docstrings (Google style, an `Args:` block).
  - FastMCP builds schemas from the signatures and docstrings.
  - Edit the descriptions in `env.py` only.
  - Exception: the paper `Agent` uses the frozen wording in `prompts/agent.json`. Do not replace it
    with API docstrings. Its fully assembled system prompts must retain the recorded hashes.
- **Paper agent:** one `Agent` in `agent.py`, no subclass or prompt-extension factories.
  - Six decision rounds, protocol repairs/guards, ledger and advisory near-duplicate check live here,
    not in the reusable tools. The finalizer and JSON repairs can add LLM calls, not tool rounds.
  - `max_tool_results=None` and `batch_search=False` preserve paper calls. The no-LLM baseline
    requests `max(20, 5*K)`; this is not a cap imposed on LLM tool calls.
- **Report parser:** `reports.prepare_report`, shared by Agent and RL scoring. Preserve ordered
  seen IDs all the way through the adapters; converting to a set changes backfilled selections.
- **Judge:** `judge.py` and `prompts/direct_relevance.json`. Post-report only, no history, gold,
  tool identity, scores or agent explanations. No automatic judge calls in a training rollout.
- **Model transport:** `model_profiles.py` and `configs/model_profiles.json` hold local serving
  controls. Never rewrite the frozen agent prompt to toggle thinking. Unknown models require an
  explicit compatible profile; hosted reasoning is provider-managed, not labeled thinking-off.
  Preserve Qwen request parity. Other profiles are documented interfaces, not evidence of recovered
  paper launch settings; do not claim live validation based on mocked tests.
- **Settings:** the `Settings` dataclass.
  - `Settings.from_environ()` reads `RECOATLAS_<FIELD>` for every field. Booleans are `"1"`/`"0"`,
    and `"none"` means `None` for numbers.
  - A new field automatically gets an environment variable. The adapters use `shared_env()`, which
    is cached, so it is one `Env` per process.
- **Datasets:** `assets.DATASETS` and `assets.SOURCES`.
  - Nothing else may be dataset-specific.
  - `catalog_items` and the split row counts are checked at load.
  - Datasets with fewer splits (Amazon Electronics only has `agent_hidden`) list only those.

## Adapter contracts (verified against the libraries' source)

- **OpenEnv** (`openenv_env.py`):
  - **The server calls `step_async`, not `step`.** Override both, as the module does.
  - **Tool coroutines run blocking tool work in `asyncio.to_thread`** so sessions don't block each
    other. `submit_report` is sync and cheap.
  - **Unknown `reset` arguments are silently ignored.** The server passes the client's keyword
    arguments to `reset` filtered by its signature, and since `reset` takes `**kwargs`, nothing is
    dropped and anything the code doesn't use is ignored. So a misspelled `candidate_id` falls back
    to a random request. New reset options must be explicit parameters.
  - **`create_app` takes the class as a factory, one instance per session.** Instances share
    `shared_env()`.
  - **Run uvicorn with 1 worker.** Each worker would load its own ~5 GB copy of the tools.
- **TRL** (`trl.py`):
  - `GRPOTrainer(environment_factory=...)` needs a class with a no-argument `__init__` and a
    `reset(**row)` that receives the dataset row's columns.
  - Every public method except `reset`/`get_reward` becomes a tool; `get_reward` is the env-owned
    reward.
  - TRL does not take OpenEnv environments directly. Its experimental OpenEnv harness is for agents
    that run their own loop, not for this.
- **Miles** (`miles.py`):
  - Matches the service interface of the RecoAtlas example in a Miles fork
    (`examples/recoatlas/agent.py`): `execute(category, tool_name, arguments)` returns
    `{"ok", "content", "product_ids", "metrics"}`, and
    `score(task_type, ground_truth, report, seen_product_ids)` returns `{"reward", "metrics"}`.
  - The Miles agent tracks seen IDs itself from `product_ids`. So only discovery tools return
    `product_ids`.
  - `miles_rows` puts K in `metadata["top_k"]`; that is Miles' name for the report size.
  - Upstream Miles' OpenEnv support is a Terminal-Bench-2-specific agent function, not a generic
    connector.

## Tried and rejected (don't redo without new evidence)

- **One container per episode (Harbor):** each parallel episode would load its own ~5 GB copy of
  the tools.
- **Pinning transformers 4.57.1:** not needed once the encoder loads in float32. Outputs are
  identical on 5.x.
- **Running TRL through the OpenEnv server:** adds a server and a network hop for no gain in one
  process. It is only useful for sharing one copy of the tools across processes or machines.
- **Sharing the search and near-duplicate product vectors (430 MB):**
  - they are normalized on different devices (CPU then moved, vs on `device`);
  - on GPU the copies differ in the last bits, so sharing would change near-duplicate results.
- **Sharing the product-text dicts between tools:** saved 31 MB measured. Not worth the code.
- **Batching by default:**
  - batch size changes GPU reductions, so results depend on what else was batched, even without
    padding;
  - measured: scores within 1e-4, and 98/100 queries with identical order (semantic);
  - so it's opt-in (`batch_search`). Searches are batched only when concurrent, with no waiting
    window.
- **Accepting `item_id` in reports:** it changes the paper's validity rule. Clearer instructions
  fixed most small-model failures instead.

## Traps

- **Each process loads the tools itself:** ~5 GB of RAM and ~6 s when files are cached.
- **Downloads are verified once.** The verification cache is `~/.cache/recoatlas/verified.json`,
  keyed by path, size and mtime. A changed file is re-hashed.
- **Speed:** CPU search takes ~180 ms and GPU ~18 ms. Prefer `device="cuda"` for anything at scale.
- **vLLM** (used in the examples) builds kernels with `ninja`. Put its venv's `bin/` on `PATH`.
- **Point temp and compile caches at a large disk** (`TMPDIR`, `TORCHINDUCTOR_CACHE_DIR`,
  `TRITON_CACHE_DIR`, `VLLM_CACHE_ROOT`, `FLASHINFER_WORKSPACE_BASE`) on machines with a small
  `/tmp`.
- **`examples/miles_agent.py` needs a Miles checkout containing `examples/recoatlas/`** (`--miles
  PATH`). That example is not in upstream Miles.

## Before making the repo public

- **There is no LICENSE file.** `src/recoatlas/tools/` is copied from recoatlas-v1, which has no
  license file either. Settle the license with the RecoAtlas authors and add a `LICENSE`.
- **`miles.py` and `examples/miles_agent.py` depend on non-public code.** They only work with the
  RecoAtlas example in an internal Miles fork (`examples/recoatlas/`), which is not in upstream
  Miles. Before going public, either say so in the README or leave these two files out.

## Tests

- **Tests cover rules the benchmark depends on:**
  - SetHit's first-K rule;
  - filtering/backfilling before rewards, and −1 for unresolved invalid reports;
  - replay parity with the original paper agent, including assembled model messages;
  - blind judge inputs and unresolved judge failures;
  - the result cap;
  - the seen rule;
  - the OpenEnv episode ending with the right reward;
  - the Miles contract.
- **Don't add tests that:**
  - mirror an implementation line by line;
  - assert a default value;
  - repeat checks the code already does at load;
  - test float tolerances that can flip on near-ties.
- **Verify parity changes with probe comparisons** (Invariant 1), not unit tests.

## Writing style for docs and messages

- Plain, precise language. State what something does and the measured number; no buzzwords and no
  hedging.
- Keep each fact in one place: `ELI5.md` is for users, `docs/overview.md` for internals,
  `README.md` is a short front page, and this file is for agents.
- Docs describe the current design. Open questions belong in issues or chat, not in the docs.

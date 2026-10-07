# How recoatlas works

What RecoAtlas is and how to use it: [ELI5.md](../ELI5.md). This document explains the internals:
where the data comes from, how the code is organized, the design choices, and measured speed and
memory.

## Data

Every file is downloaded from Hugging Face at a pinned revision and checked against a checksum. The
pins come from the dataset's entry in recoatlas-v1's `artifact_sources.json`.

- **Requests:** [`iaouali/hm-benchmark`](https://huggingface.co/datasets/iaouali/hm-benchmark).
  - Splits: `shared_train` (9,019), `agent_dev` (2,028), `agent_hidden` (1,960).
  - Each request has a text query and 3–7 reference product IDs; K is their number.
- **Tools:** [`iaouali/hm-tools`](https://huggingface.co/iaouali/hm-tools): product texts,
  subcategories, product vectors, complementarity vectors, and a LoRA fine-tune for search.
- **Query encoder:** `Qwen/Qwen3-Embedding-0.6B`, which turns search queries into vectors.

## Tool code

The tool code in `src/recoatlas/tools/` is copied from the RecoAtlas release,
[recoatlas-v1](https://github.com/imadaouali/recoatlas-v1). Three changes were made to it:

- **Imports and file paths,** adjusted to this package.
- **The query encoder loads in float32.** v1 got float32 by default. transformers 5 defaults to
  the checkpoint's bfloat16, and that changes which products search returns.
- **`search_many`** encodes several queries in one pass; it is used by `batch_search`. A single
  search runs through the same code.

The tools return the same JSON as v1 for the same inputs. This was checked on 300 calls per tool
variant, under both transformers 4.57.1 and 5.19.

## Code

| module | role |
|---|---|
| `assets.py` | pinned files and checksums per dataset (`DATASETS`, `SOURCES`); download, verify, load splits |
| `env.py` | `Settings`; `Env` (tools and requests, loaded once); `Episode` (one request); the agent-facing tool descriptions (`TOOL_DOCS`) |
| `scoring.py` | `sethit`, label-free report validation, `score_report` |
| `batching.py` | `SearchBatcher`, used when `batch_search=True` |
| `baselines.py` | v1's no-LLM baseline, plus `run_baseline(episode)` |
| `openenv_env.py`, `trl.py`, `miles.py` | the three adapters |

**`Env`:**
- It loads the tools once, in about 6 s when the files are cached.
- Its tool methods keep no state between calls and are thread-safe, so one `Env` serves any number
  of concurrent episodes.

**`Episode`:**
- It records the products returned by `search_products` and `get_complementary_products` (the
  "seen" products, as in v1's candidate ledger) and every submitted report.
- `submit()` checks a report without using the reference items. Submitting again replaces the
  report.
- `score()` scores the last report.

**Scoring:**
- **`sethit`** cuts the submitted list to its first K positions before removing duplicates and
  unknown IDs, as in v1's `evaluation/metrics.py`, so padding a report earns nothing.
- **`reward_direct`** is `sethit` for a valid report and −1 otherwise. This is the training reward
  of the RecoAtlas paper.
- **The report format is strict:** `[{"product_id": "...", "reasoning": "..."}, ...]`. Small models
  often write `item_id`, the key the tools return. So the validation error names the keys it found,
  and the TRL instructions spell out the format.

## Design choices

**A Python package, not a container format:**
- **No sandbox is needed.** The tools are pure functions over fixed files and never run code the
  agent writes.
- **One copy of the tools.** The tools take about 5 GB of memory (see below), so they are loaded
  once per process and shared by every episode.
- **One container per episode would multiply that copy.** That's the model of formats like Harbor:
  128 parallel RL episodes would hold 128 copies.

**`max_tool_results` (default 20) is the only behaviour added to v1, which has no limit:**
- the validity rule only accepts products that a tool returned;
- without a limit, one search for 100,000 products would make every product "seen" and fill the
  model's context;
- a larger `top_k` is cut to the limit, and `None` removes it.

**Settings come from environment variables in the adapters:** `RECOATLAS_<SETTING>` (e.g.
`RECOATLAS_DEVICE=cuda`), plus `RECOATLAS_DATASET`. See `Settings.from_environ`.

**Three adapters, each a thin wrapper around the same `Env` and `Episode`,** so the tools and the
scoring exist once:

- **OpenEnv** (`openenv_env.py`):
  - one [OpenEnv](https://github.com/huggingface/OpenEnv) session is one episode;
  - the three tools and `submit_report` are MCP tools, and `submit_report` ends the episode with
    reward `reward_direct`, also for an invalid report;
  - `recoatlas-openenv` serves all sessions from one process. Tool calls run in worker threads, so
    one session's search doesn't block the others.
- **TRL** (`trl.py`):
  - `GRPOTrainer` does not accept an OpenEnv environment; it takes an `environment_factory`, a class
    with a no-argument constructor, `reset`, the tools and `get_reward`;
  - this module is that class, running in the training process.
- **Miles** (`miles.py`):
  - `MilesToolService` implements the `execute`/`score` interface of a Miles RecoAtlas agent loop,
    and `miles_rows` builds its data rows;
  - in a Miles job it runs as one Ray actor.
  - Miles' own OpenEnv support is an example written for one other environment
    (Terminal-Bench-2), so it can't be reused here.

**Each dataset is one entry in `assets.py`.** No other code is dataset-specific. See "Adding a
dataset".

## Memory

Measured after loading on CPU:

| what | size |
|---|---|
| query encoder: 616M parameters in float32 | 2,464 MB |
| product vectors for search (104,993 × 1,024 floats) | 430 MB |
| complementarity "anchor" vectors | 430 MB |
| complementarity "complement" vectors | 430 MB |
| product vectors for the near-duplicate tool | 430 MB |
| subcategory masks | 27 MB |
| **tensors in total** | **4,211 MB** |
| product texts, Python and library overhead | about 800 MB |

**The near-duplicate tool's vectors are a second copy of the search vectors.** They stay separate
because the two tools normalize them on different devices. On GPU that gives copies that differ in
the last bits, so sharing one would change near-duplicate results.

With `device="cuda"`, the tensors are in GPU memory.

## Speed

- **One search:** about 18 ms on an NVIDIA B200 and about 180 ms on CPU. The other two tools only
  multiply precomputed vectors and are cheaper.
- **The no-LLM baseline on all 1,960 `agent_hidden` requests:**

| setup | time |
|---|---|
| 32 threads | 33.4 s |
| 32 threads, `batch_search=True` | 7.6 s |
| OpenEnv server, 64 sessions, `batch_search=True` | 41 s |

### Search batching (`batch_search`)

Search time is almost all the 0.6B encoder, and encoding 32 queries together costs about the same
as encoding one.

- **How it works:** one worker thread owns the encoder.
  - A search that arrives while the encoder is idle runs at once; there's no waiting window.
  - Searches that arrive while a batch is running all go into the next batch, up to 64.
- **Only `search_products` is batched.**
- **When it helps:** only when searches arrive at the same time, from threads, server sessions or
  parallel Miles rollouts. A single episode on its own gains nothing.
- **Why it is off by default:** GPUs compute slightly different floats for different batch sizes,
  even without padding. So with batching on, a search's result depends on which searches shared
  its batch. Measured on 100 queries:
  - scores shifted by up to 0.0001;
  - products came back in the same order for 100/100 (utility) and 98/100 (semantic) queries; the
    two exceptions were swaps between nearly tied products.

  Off gives exact results, reproducible and identical to v1.

## Checked results (H&M)

- **No-LLM baseline on all 1,960 `agent_hidden` requests:** 0.70% `sethit` with the utility tools
  and 0.49% with the semantic tools; every report valid. The same 0.70% through the OpenEnv server.
- **Miles:** Miles' RecoAtlas agent loop with Qwen3-4B on 400 requests:
  - every report valid, 0.9% `sethit`;
  - every reward matched when recomputed independently by this package.
- **TRL:** 3 GRPO steps on Qwen3-0.6B with no tool failures.

## Adding a dataset

To add an Amazon dataset from the RecoAtlas release (e.g. Musical Instruments):

1. **Add the source repos.** Put `iaouali/amazon-benchmark` and `iaouali/amazon-tools`, with their
   pinned revisions, in `SOURCES`.
2. **Add the dataset.** Put a `DATASETS` entry under the dataset's name, containing:
   - `catalog_items`;
   - `splits`: the published splits with their row counts;
   - `files`: the same file names as H&M's, each with its source, path, size and checksum.

   All values come from that dataset's entry in recoatlas-v1's `artifact_sources.json`.
3. **Load it:** `recoatlas.load("amazon-musical-instruments")`.

Each dataset lists only the splits it publishes; Amazon Electronics, for example, only has
`agent_hidden`.

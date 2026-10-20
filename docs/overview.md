# How recoatlas works

What RecoAtlas is and how to use it: [ELI5.md](../ELI5.md). This document explains the internals:
where the data comes from, how the code is organized, the design choices, and measured speed and
memory.

## Data

Every file is downloaded from the unified
[`iaouali/recoatlas`](https://huggingface.co/datasets/iaouali/recoatlas) repository at one pinned
revision and checked against a SHA-256 checksum. Query IDs, partitions, text, reference order and
the five frozen serving artifacts per environment are preserved from the audited source releases.

- **H&M Clothing:** 104,993 products; `shared_train` (9,019), `agent_dev` (2,028), and
  `agent_hidden` (1,960). Each request has a text query and 3–7 reference product IDs; K is their
  number.
- **Amazon Musical Instruments:** 62,961 products; `shared_train` (8,937), `agent_dev` (2,028),
  and `agent_hidden` (1,960). The published training file has 8,937 rows: 9,019 were generated and
  82 were removed by the final overlap filter.
- **Tools in each environment:** product texts, subcategories, product vectors, complementarity
  vectors, and the learned query-side retrieval checkpoint.
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
| `agent.py` | single frozen three-tool `Agent`, including protocol repair and advisory near-duplicate guidance |
| `prompts/agent.json`, `configs/paper_agent.json` | frozen assembled prompt components and paper run settings |
| `reports.py` | original report parser, shared by the agent and all RL reward paths |
| `scoring.py` | `sethit`, post-processing validity and RL reward |
| `backend.py`, `output_contract.py` | compatible inference transport, thinking settings, compact three-tool JSON grammar |
| `model_profiles.py`, `configs/model_profiles.json` | model-specific chat-template controls and required reasoning parser/boundaries; no prompt edits |
| `run_agent.py` | saved trajectories and post-report exact recovery; no judge calls |
| `run_store.py` | locked, atomic compressed query checkpoints; strict resume identity and ordered JSONL export |
| `sweep.py`, `configs/open_models.json` | pinned open-model sweep, local vLLM lifecycle, development gates, sharding and time limit |
| `judge.py`, `prompts/direct_relevance.json` | blind whole-set direct-relevance judgments and saved attempts |
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
- `submit()` runs the same filtering/backfilling parser as the paper agent and returns the
  processed report with label-free validity. Raw submissions remain available for auditing.
  Submitting again replaces the report being scored.
- `score()` scores the last report.

**Scoring:**
- `reports.prepare_report` rejects malformed result schemas, filters duplicate and out-of-catalog
  IDs, and filters unobserved IDs when the candidate list is nonempty. It fills short reports from
  observed candidates in their discovery order, then truncates to K. Backfilled items are marked
  by their reason string. The parser never reads the reference set.
- For exact replay fidelity, the original empty-candidate edge case is retained: with no candidates,
  catalog IDs are not filtered by observation. RL validity still rejects unobserved selections.
- **`sethit`** cuts the submitted list to its first K positions before removing duplicates and
  unknown IDs, as in v1's `evaluation/metrics.py`. Here it receives the processed report.
- **`reward_direct`** is `sethit` for a valid processed report and −1 otherwise. The adapters keep
  this reward interface, but validity is determined after the paper parser; there is no alternate
  strict-raw-report scoring mode. This packaging choice should not be described as a replay of
  historical RL training runs that used strict raw validity.
- **The report contract** is `[{"product_id": "...", "reasoning": "..."}, ...]`; `item_id` is not
  accepted as an alias. The grammar requires reasoning, while the original parser tolerates its
  absence and supplies a fallback reason. Other malformed schemas fail rather than being backfilled.

**Paper agent and RL interfaces:**
- `Agent` owns the six-round budget, minimum-evidence repair/guard behavior and persistent ledger.
  There is no inheritance chain or separate diversity agent. Guard calls consume tool rounds;
  repair requests and finalization are model calls but not extra tool rounds.
- Near-duplicate checks use 1–60 unique previously observed IDs at threshold 0.95. Invalid check
  arguments return an error as that round's observation. Valid checks leave the candidate pool
  intact. Actual use is logged, not forced.
- Generic RL policies still use the tools through OpenEnv, TRL or Miles. Their native function-call
  prompts/loops are owned by the trainer; sharing the environment does not make those loops exact
  replays of `Agent`. Use `recoatlas-agent` for the frozen paper-agent inference protocol.
- The judge is a post-report measurement, not a tool or automatic RL reward. It judges each report
  independently. Missing judge responses are unresolved measurements; summaries include coverage.

**Parity checks:** `tests/fixtures/paper_parity.json` records 15 synthetic replay cases against
the original runner at commit `5e818e6ee4766dd1ec5aa646cc39b1d6dbff6aaf`. Tests compare report hashes,
every model request excluding elapsed-time-dependent timeouts, tool calls and protocol counters.
They cover K=3–7, the finalizer, repair/guard actions, optional/invalid near-duplicate checks,
backfilling, malformed reports and the ledger limit. Five system-prompt hashes independently pin
the fully assembled prompts. These tests verify implementation parity, not fresh model outcomes.

**Serving profiles:** `model_profiles.resolve_profile` selects by recognized model identifier or
explicit profile for served aliases. Unknown local models and conflicting profiles fail before
tools load or the client sends a request. `ChatBackend` forwards the profile's chat-template
controls together with the original compact grammar; the model messages are unchanged. Profiles
share the paper configuration's temperature and output/thinking limits. Server parser/boundary
requirements and provenance are stored with each run, not asserted as remotely verified settings.
Qwen's original requests are preserved. `configs/open_models.json` identifies all four model pins,
context lengths, reasoning settings and the recorded serving build; remaining server arguments are
explicit launcher choices. See the user guide for sources and
serving setup. Mock request/response tests cover both thinking modes, alias handling, grammar
coexistence, separate reasoning fields, rejected mode mismatches and CLI preflight. They do not
replace testing the real models or verifying the server's budget enforcement.

## Design choices

**A Python package, not a container format:**
- **No sandbox is needed.** The tools are pure functions over fixed files and never run code the
  agent writes.
- **One copy of the tools.** The tools take about 5 GB of memory (see below), so they are loaded
  once per process and shared by every episode.
- **One container per episode would multiply that copy.** That's the model of formats like Harbor:
  128 parallel RL episodes would hold 128 copies.

**Candidate counts:** the default `max_tool_results=None` forwards the requested count unchanged.
The no-LLM baseline requests `max(20, 5*K)`. LLM requests are bounded by the frozen grammar to 1–100,
not by the baseline's count. An optional environment cap remains available for custom RL experiments;
the frozen paper agent refuses a capped environment.

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
- **The no-LLM baseline on all 1,960 `agent_hidden` requests**, measured with an explicit
  `max_tool_results=20` cap (not the paper-agent setting):

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

## Adding a dataset

To add another dataset from the RecoAtlas release:

1. **Add any source repos not already registered,** with pinned revisions in `SOURCES`.
2. **Add the dataset.** Put a `DATASETS` entry under the dataset's name, containing:
   - `catalog_items`;
   - `splits`: the published splits with their row counts;
   - `files`: the same file names as H&M's, each with its source, path, size and checksum.

   All values come from that dataset's entry in recoatlas-v1's `artifact_sources.json`.
3. **Load it** with `recoatlas.load(...)`.

Each dataset lists only the splits it publishes; Amazon Electronics, for example, only has
`agent_hidden`.

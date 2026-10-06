# RecoAtlas

Run a recommendation agent with the paper's frozen bundle-shopping prompt,
catalog tools, and evaluation protocol. The agent searches for products,
expands complementary candidates, checks near duplicates, and returns a set.
Evaluation then provides fractional **SetHit@K** and, optionally, a separate
**LLM-as-a-judge direct-relevance score**.

The agent sees the shopping request, requested set size, and tool observations.
It **never sees reference items, SetHit scores, or judge feedback while choosing
products**. Feedback is computed after the final report; it is not used for
policy updates.

This repository contains the agent/evaluation runtime, not RL training,
dataset construction, tool training, cluster launchers, or figure-reproduction
pipelines. Model weights and benchmark data are downloaded separately.

## 1. Requirements

- Python 3.11 or 3.12; the release is tested on Linux with Python 3.12.
- Internet access for installation and the first artifact/model downloads.
- CPU and enough RAM for catalog embeddings and the 0.6B query encoder, or a
  CUDA device for faster tools. Allow roughly 16 GB of system RAM for CPU use.
- For an **LLM agent**, a running Chat Completions-compatible model endpoint
  and its served model name. This code is a client; it does not start a model
  server or include the 27B agent weights. A hosted endpoint also works.
- For **LLM judging**, access to a judge endpoint and its model/deployment name.
  Agent and judge endpoints can be different.

No API key is needed for the deterministic baseline or local SetHit evaluation.
The pinned HF assets are public, so no HF token is required to download them.
Hosted LLM calls generally require a provider API key and may incur charges.
An unauthenticated local server can use `OPENAI_API_KEY=EMPTY`.

## 2. Install

```bash
git clone https://github.com/imadaouali/recoatlas-v1.git
cd recoatlas-v1
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# CPU-only tools (omit this line if installing a CUDA-enabled PyTorch build).
python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pytest -q
```

For the paper's grammar-constrained Qwen agent, also install:

```bash
python -m pip install -r requirements-grammar.txt
```

`requirements-grammar.txt` pins the grammar compiler. The agent endpoint must
accept `structured_outputs.grammar`; installing the client library alone does
not enable grammar enforcement on the server. The runtime dependencies are
separate from the model-server environment.

## 3. Download a benchmark category

```bash
python prepare_artifacts.py --category amazon-musical-instruments
python scripts/verify_artifacts.py --category amazon-musical-instruments
```

The first command downloads the catalog, subcategories, cached embeddings,
selected retrieval adapter, and available query/reference files into
`artifacts/amazon-musical-instruments/`. Only inference assets are fetched;
optimizer experiments and checkpoint histories are not downloaded separately.
Every file is checked against a pinned commit and checksum in
[`artifact_sources.json`](artifact_sources.json). The original retrieval
checkpoint may itself contain training metadata; loading uses `weights_only=True`.

| Category argument | Published hidden queries | Status |
| --- | ---: | --- |
| `amazon-musical-instruments` | 1,960 | Agent and evaluation available |
| `amazon-electronics` | 339 | Agent and evaluation available |
| `hm-clothing` | 1,960 | Agent and evaluation available |
| `amazon-video-games` | Not available in the pinned publication | Tools only; benchmark evaluation blocked |

Use the same commands with another category argument. Video Games' tools can
be downloaded with `--tools-only`, but its original query/reference file is
not in the pinned publication. The runner does not reconstruct gold IDs from
predictions or substitute another dataset.

Assets currently come from the existing `iaouali/amazon-benchmark`,
`iaouali/hm-benchmark`, `iaouali/amazon-tools`, and `iaouali/hm-tools` repositories.
The proposed consolidated HF repositories are not required by this release.
See [artifact details](docs/data.md) for exact layouts, counts, and limitations.

Approximate category downloads are 0.67 GB (Musical Instruments), 0.85 GB
(Electronics), and 1.59 GB (H&M). Retrieval additionally downloads the pinned
`Qwen/Qwen3-Embedding-0.6B` backbone on first use. Allow extra disk space because
the Hugging Face cache and the local artifact copies are separate.

To use a different artifact location:

```bash
python prepare_artifacts.py --category hm-clothing --output /path/to/artifacts
export RECO_ATLAS_ARTIFACTS=/path/to/artifacts
```

All run/evaluation commands also accept `--artifacts /path/to/artifacts`.
Subsequent inference can run without HF network access after both the artifacts
and query encoder have been cached; set `HF_HUB_OFFLINE=1` for that check.
The LLM endpoint must still be reachable.

## 4. Check the tools without an LLM or API key

```bash
python generate_traces.py \
  --category amazon-musical-instruments \
  --agent non-llm --tools utility --device cpu --limit 1 \
  --output outputs/baseline-smoke.jsonl
```

This runs the paper's retrieval + complementarity + near-duplicate baseline
on one query. It writes predictions and per-query SetHit feedback to JSONL,
plus `outputs/baseline-smoke.summary.json`. A one-query score is only a smoke
test, not the paper's aggregate baseline result. Use `--device cuda` for GPU
tools, or `--tools semantic` for pretrained-embedding tools.

## 5. Configure and run the LLM agent

First start or obtain access to your model endpoint. For the paper's Qwen
configuration it must serve `Qwen/Qwen3.8-27B-FP8` at the revision recorded in
[`configs/paper_agent.json`](configs/paper_agent.json), support the Qwen chat
template controls, and accept the grammar field described above. The client
records the requested model but cannot verify the weights actually served.

Set the endpoint and authentication in your shell, not in tracked files:

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=EMPTY  # Only for an unauthenticated local server.
```

For a hosted endpoint, replace those values with the provider's base URL and
your key. The code reads environment variables; it does **not** automatically
load a `.env` file. Never commit credentials.

Start with one query, thinking disabled:

```bash
python generate_traces.py \
  --category amazon-musical-instruments \
  --agent llm --tools utility --thinking off --device cpu --limit 1 \
  --output outputs/qwen-utility-off-smoke.jsonl
```

Then run all hidden queries by omitting `--limit`:

```bash
python generate_traces.py \
  --category amazon-musical-instruments \
  --agent llm --tools utility --thinking off --device cuda \
  --output outputs/qwen-utility-off.jsonl

python generate_traces.py \
  --category amazon-musical-instruments \
  --agent llm --tools semantic --thinking off --device cuda \
  --output outputs/qwen-semantic-off.jsonl
```

Both conditions use the **same frozen agent prompt**, six tool-decision
rounds, requested set size, and near-duplicate tool. Only retrieval and
complementarity change. `--device` controls local catalog tools, not the LLM
server. The default split is `agent_hidden`; `--partition agent_dev` is
available where that partition is published.

Thinking-on uses `--thinking on`. The paper settings are temperature 1,
1,000 output tokens without thinking, or 16,384 output tokens with a
4,096-token thinking budget. Thinking-on requires a Qwen-compatible endpoint
that separates reasoning from final content and honors `thinking_token_budget`.
An endpoint that ignores those options does not implement that condition.

To use a different model or an endpoint without grammar support, pass
`--model YOUR_SERVED_MODEL --no-grammar`. This is a supported execution mode,
**not** a claim to reproduce the paper's Qwen serving configuration. Token
limit overrides are likewise recorded as run settings.

Runs are sequential and flush one trace per query. Existing output/summary
files are never overwritten. There is no automatic resume; retain interrupted
traces, use a fresh output path for another run, and do not concatenate duplicate
query IDs. Errors are recorded with their exception type, and a run with failed
queries exits nonzero. Model calls may be retried, so API requests can outnumber
tool rounds. Do not launch full evaluation before checking the one-query output.

## 6. Evaluate SetHit@K

```bash
python evaluate_traces.py \
  --category amazon-musical-instruments \
  --traces outputs/qwen-utility-off.jsonl
```

SetHit@K is the fraction of reference items recovered in the first K report
positions, where K is the reference-set size (3–7). It is **not** binary
any-overlap. Duplicate and invalid positions earn no extra credit.

The standalone evaluator uses the **entire selected partition** as denominator;
missing/failed reports score zero. The summary written during generation uses
only the queries selected by `--limit`. Thus evaluating a one-query smoke file
against the full split intentionally yields a different denominator. Both
summaries label expected/completed query counts and completed-only scores.

## 7. Run the direct-relevance LLM judge

The judge receives the request, K, and anonymously labeled catalog descriptions
of the recommended products. It does not receive the reference set, purchase
history, tool/model identity, tool scores, SetHit, or agent explanations.
The fixed [rubric](prompts/direct_relevance.json) returns **one integer score
for the entire set**, from 0 to 4, normalized by dividing by four.

Validate inputs without making any API calls:

```bash
python judge_traces.py \
  --category amazon-musical-instruments \
  --traces outputs/qwen-utility-off.jsonl --dry-run
```

Then configure the judge's endpoint/key (separately from the agent, if needed):

```bash
export OPENAI_BASE_URL=https://YOUR_JUDGE_ENDPOINT/v1
export OPENAI_API_KEY=YOUR_JUDGE_API_KEY

python judge_traces.py \
  --category amazon-musical-instruments \
  --traces outputs/qwen-utility-off.jsonl \
  --model YOUR_JUDGE_DEPLOYMENT \
  --output outputs/qwen-utility-off.judge.jsonl
```

`gpt-5.6-luna` is the paper's managed deployment alias, **not a promise that
this identifier is available from a public API**. Supply it only if your endpoint
provides that deployment. A different model means a different evaluator; the
summary records the requested model and original alias. Matching an alias alone
does not prove matching weights. Agent API keys do not automatically grant
access to the judge service.

Repeat with the semantic trace file to compare conditions. Each complete report
is judged independently, including identical reports in different runs. At most
three judgment attempts are made, in addition to transport retries. Invalid
or missing judgments are unresolved measurements, not zero-quality scores.
The summary reports valid judgments, missing coverage, and mean direct relevance.
For paired analysis, compare the same successfully judged query IDs; do not
compare silently different subsets.

## What is frozen, and what is verified?

- [Strategy](prompts/strategy.txt), agent scaffold/finalizer, diversity guidance,
  and JSON output contract from the paper implementation.
- [Agent settings](configs/paper_agent.json) and the direct-relevance judge's
  exact system instruction, rubric, shuffle seed, and decoding settings.
- [Artifact source revisions and checksums](artifact_sources.json), including
  a pinned query-encoder backbone revision.
- Focused regression tests for protocol behavior, gold-blind requests, SetHit,
  invalid reports, artifact checksums, and the agent-to-judge workflow.

See [protocol details](docs/protocol.md) and [validation scope](docs/validation.md).
Fresh stochastic LLM runs need not reproduce the paper's numerical outcomes.
No saved paper-result table is substituted for live evaluation.

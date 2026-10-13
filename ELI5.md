# recoatlas

## What it is

RecoAtlas is a benchmark and RL environment for LLM recommendation agents. One episode works like
this:

1. **The agent gets a shopping request and a number K.** For example: *"I'm putting together a
   polished casual summer outfit with a comfortable short-sleeved top, relaxed high-waisted shorts,
   and a statement waist accent to define the silhouette."* K is how many products it must
   recommend.
2. **It searches a fixed catalog of 104,993 H&M products with three tools:**
   - `search_products(query, top_k)`: the `top_k` products closest to a text query;
   - `get_complementary_products(item_ids, top_k)`: products that complement the given products,
     from other subcategories (learned from co-purchases in the default "utility" tools);
   - `get_substitute_products(item_ids)`: splits a list into kept products and removed
     near-duplicates (same subcategory, very similar).
3. **It submits a report** of K products, each with a short reason. The paper parser removes
   duplicate, unknown and unobserved IDs, then fills missing positions from earlier discoveries.
4. **The report is compared with the request's reference items:** K products taken from one real
   customer's purchases. The request was written to describe those products. K is between 3 and 7.

## The score

- **`sethit`:** the share of the reference items found among the first K submitted products, from
  0 to 1.
- **`valid`:** 1 if the report has exactly K different catalog products, each returned earlier by
  `search_products` or `get_complementary_products`. Otherwise 0.
- **`reward_direct`:** `sethit` if valid, otherwise −1. This is the RL reward.

Validity and rewards are computed **after** filtering/backfilling, in every RL adapter.
Backfilling uses discovery order, never reference labels. Malformed schemas fail without
backfilling. If too few candidates were discovered, the processed report can remain short;
it receives partial SetHit but an invalid-report RL reward. A missing/failed report has zero
SetHit and reward −1. SetHit and the training reward are separate signals.

## 1. Install

From a clone of this repo:

```bash
uv sync --extra tools
```

## 2. Load the environment

```python
import recoatlas

env = recoatlas.load("hm-clothing", device="cuda")  # or device="cpu"
```

- **First call:** downloads about 2.8 GB into the Hugging Face cache and checks every file.
- **Later calls:** about 6 seconds.
- **Memory:** about 5 GB of RAM.

## 3. Run one episode by hand

```python
ep = env.episode(split="agent_hidden", index=0)
print(ep.request)  # the shopping request
print(ep.k)  # how many products to submit

ep.search_products("relaxed high-waisted shorts", top_k=10)  # returns JSON text
ep.get_complementary_products(["0722674001"], top_k=5)

picks = ep.seen_ids[: ep.k]  # products the tools have returned so far
print(ep.submit([{"product_id": p, "reasoning": "fits the outfit"} for p in picks]))
print(ep.score())  # {'sethit': ..., 'valid': 1, 'reward_direct': ...}
```

- **`submit` returns the processed report** under `report`, plus label-free `valid` and `errors`.
  Raw submissions are retained in `ep.submissions`. Rewards and references are not returned by `submit`.
- **You can submit again.** The last report is the one scored.

## 4. Run a whole split

Every request in a split, with the built-in no-LLM baseline:

```python
from concurrent.futures import ThreadPoolExecutor
from recoatlas.baselines import run_baseline

n = len(env.splits["agent_hidden"])  # 1,960 requests
with ThreadPoolExecutor(32) as pool:
    scores = list(pool.map(lambda i: run_baseline(env.episode("agent_hidden", index=i)), range(n)))
print(sum(s["sethit"] for s in scores) / n)
```

All threads share one `env`, so the tools are loaded once.

## 5. Run the paper agent

Install the inference client, tools and frozen grammar compiler:

```bash
uv sync --extra agent --extra grammar
```

You need a running compatible model endpoint. The paper configuration is packaged in
`src/recoatlas/configs/paper_agent.json`; it records the Qwen checkpoint/revision, temperature and
thinking limits. Local vLLM serving profiles cover Qwen, Gemma 4, Nemotron 3.5 Lightning and
GLM-4.7-Flash. The server must support compact `structured_outputs.grammar`, the selected model's
chat-template parameters, and (for thinking on) `thinking_token_budget` with the matching reasoning
parser. This package does not start or download the agent model. Loading `Env` downloads only
the recommendation-tool model and artifacts.

Configure the endpoint and, if it requires authentication, a key:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8001/v1
export OPENAI_API_KEY=EMPTY  # only for an unauthenticated local endpoint
```

Use your actual key for an authenticated endpoint; do not commit it. `.env` files are not loaded
automatically. Start with one development query:

```bash
uv run recoatlas-agent --dataset hm-clothing --split agent_dev --limit 1 \
  --device cuda --thinking off --output runs/smoke.jsonl
```

Then evaluate the hidden split (omit `--limit` to include every query):

```bash
uv run recoatlas-agent --dataset hm-clothing --split agent_hidden \
  --device cuda --thinking off --tools utility --output runs/utility-off.jsonl
uv run recoatlas-agent --dataset hm-clothing --split agent_hidden \
  --device cuda --thinking off --tools semantic --output runs/semantic-off.jsonl
```

Use `--device cpu` without a GPU; this controls the recommendation tools, not the remote agent.
Use `--model` to select the served model and `--thinking on` for reasoning. Both modes send an
explicit thinking switch; they do not rely on the server's default. The output limit is 1,000
tokens with thinking off and 16,384 with thinking on, including a 4,096-token reasoning budget
in the latter. Temperature is 1 in both modes. These limits are shared across the local profiles.

### Choose the agent model

The following names select their profiles automatically (case-insensitive). For another served
alias, supply `--model-profile` explicitly. A recognized model with the wrong profile is rejected.
Unrecognized local models are rejected before downloading tools or making requests; selecting a
profile for an alias asserts that its underlying model supports that interface.

| Model identifier | Profile | vLLM `--reasoning-parser` |
|---|---|---|
| `Qwen/Qwen3.8-27B-FP8` | `qwen` | `qwen3` |
| `google/gemma-4-31B-it` | `gemma4` | `gemma4` |
| `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | `nemotron` | `nemotron_v3` |
| `zai-org/GLM-4.7-Flash` | `glm4_7` | `glm45` |

The Qwen profile also recognizes the paper's Qwen3.5 size variants and Qwen3-4B/8B.
An explicit profile is not a compatibility guarantee for other model generations or quantizations.

Inspect the resolved controls without an API key, downloads, files or inference:

```bash
uv run recoatlas-agent --model google/gemma-4-31B-it --thinking on --dry-run
```

The output includes `required_server.reasoning_parser` and `required_server.reasoning_config`.
Use those in your vLLM server's `--reasoning-parser` and `--reasoning-config` options. For example,
on hardware large enough for Gemma 4 (adjust parallelism and context capacity to your hardware):

```bash
vllm serve google/gemma-4-31B-it --port 8001 --max-model-len 32768 \
  --reasoning-parser gemma4 \
  --reasoning-config '{"reasoning_start_str":"<|channel>","reasoning_end_str":"<channel|>"}'
```

Install a vLLM build that supports that checkpoint, parser, thinking budget and structured-output
combination; the serving installation is separate from this package's client dependencies. The
model's chat template must accept `enable_thinking`. See the linked official sources below for
model-specific installation requirements. This agent emits JSON actions, **not native function
calls**: do not replace its prompt or enable a different tool-calling agent loop.

Run one development query against the server, then repeat with `--thinking off` and a new output:

```bash
uv run recoatlas-agent --model google/gemma-4-31B-it --thinking on \
  --split agent_dev --limit 1 --output runs/gemma-on-smoke.jsonl
```

If the server uses a custom alias, e.g. `--served-model-name my-local-model`, use:

```bash
uv run recoatlas-agent --model my-local-model --model-profile gemma4 --thinking off \
  --split agent_dev --limit 1 --output runs/gemma-off-smoke.jsonl
```

Inspect the saved model calls: reasoning should be separate from final JSON. Responses with
unparsed reasoning delimiters, missing final content, exhausted thinking output limits, or
nonempty reasoning while thinking is off are rejected. Absence of reasoning in a particular
thinking-on response is not itself an error. Client tests do not prove that a server honors the
budget: validate both modes on the actual serving installation before running a full split.

**Provenance:** Qwen request controls retain the original release behavior. The other profiles
implement documented interfaces from [vLLM's Gemma guide](https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html),
the [Nemotron model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16),
and the [GLM model card](https://huggingface.co/zai-org/GLM-4.7-Flash) and
[chat template](https://huggingface.co/zai-org/GLM-4.7-Flash/blob/main/chat_template.jinja).
Budgeting uses [vLLM's thinking-budget interface](https://docs.vllm.ai/en/latest/features/reasoning_outputs/#thinking-budget-control).
The original non-Qwen paper launch configurations/revision pins were not recovered and these
profiles have not been live-inference validated. They provide compatibility, not evidence of
exact reproduction of the published figure. Keep the checkpoint/tokenizer revisions, vLLM
version, launch arguments and server sampling defaults with your run; these are not discovered
or verified by the client.

### Hosted endpoints and saved results

A hosted endpoint can use `--hosted --json-only --model YOUR_MODEL`. This omits local controls:
reasoning remains provider-managed (`thinking: null` in the run summary), even with the default
`--thinking off`. `--hosted --thinking on` and hosted model-profile overrides are rejected.
The frozen strategy and task contract stay fixed.

Each run writes reports, tool traces, model calls, protocol counts and post-report SetHit to JSONL,
plus a `.summary.json` containing the resolved model profile, client controls and required server
settings. The Qwen revision is not attributed to other models. Failed queries stay in the denominator
with zero SetHit. Existing files
are never overwritten. `--limit` results are subset results, not full-benchmark results.

The Python entry point is `Agent(env, backend).run(request, k)`; there is only one agent
implementation. Six decision rounds, two distinct searches and one complementarity call are
agent-side requirements. The near-duplicate call at 0.95 is requested but not forced. Its result
is advisory: it does not delete candidates or override the final selection. The environment itself
does not impose the agent's budget on arbitrary RL policies.

## 6. Judge completed reports

The direct-relevance judge gives **one 0–4 score per recommended set**, normalized by four.
It sees the request, K and anonymized catalog descriptions (300 characters each), not reference
items, purchase history, agent explanations, tool condition, retrieval scores or SetHit.

First validate saved predictions without calling an API:

```bash
uv run --extra llm recoatlas-judge --traces runs/utility-off.jsonl --dry-run
```

Then set `OPENAI_BASE_URL` and `OPENAI_API_KEY` for your judge endpoint (which can differ from the
agent endpoint), and explicitly select an accessible deployment:

```bash
uv run --extra llm recoatlas-judge --traces runs/utility-off.jsonl \
  --model YOUR_JUDGE_DEPLOYMENT --output runs/utility-off-judge.jsonl
```

The paper deployment alias is `gpt-5.6-luna`; using another model is recorded as such. For traces
from `agent_dev`, add `--split agent_dev`. Judging incurs API charges and is never launched
automatically by an agent rollout or reward computation. Up to three attempts are recorded per
report; the first valid response is used. Failed judgments remain unresolved, not zero. Failed
agent reports are skipped; an incomplete report marked complete is rejected during preflight.
Judging requires only the `llm` extra and catalog/query files, not embeddings or a GPU.

## 7. Plug in your own agent

For each request, your agent:

1. reads `ep.request` and `ep.k`;
2. calls tools with `ep.call(tool_name, **arguments)`. The tool descriptions, for a prompt or a
   tool schema, are in `recoatlas.env.TOOL_DOCS`;
3. finishes with `ep.submit(results)`.

Then average `ep.score()["sethit"]` over the split.

## 8. Train with RL

- **TRL:** pass `recoatlas.trl.RecoAtlasEnvironment` to `GRPOTrainer` as `environment_factory`.
  See `examples/trl_grpo.py`.
- **OpenEnv-based trainers:** start the server, one process for all episodes:

  ```bash
  uv sync --extra server
  RECOATLAS_DEVICE=cuda uv run recoatlas-openenv --port 8000
  ```

- **Miles:** `recoatlas.miles.MilesToolService` is the tool service for a Miles RecoAtlas agent
  loop. See `examples/miles_agent.py`.

## Settings

Pass them to `recoatlas.load(...)`. The adapters read them from `RECOATLAS_<NAME>` environment
variables instead, e.g. `RECOATLAS_DEVICE=cuda`.

| setting | default | what it does |
|---|---|---|
| `tools` | `"utility"` | `"utility"`: search and complementary tools fine-tuned on purchase data. `"semantic"`: the pretrained embeddings without that fine-tuning. |
| `max_tool_results` | `None` | optional cap for custom experiments; the paper agent requires no environment cap |
| `tool_noise` | `0.0` | share of complementary and near-duplicate results deliberately corrupted (by default, swapped for a random product of the same subcategory), to test robustness to faulty tools |
| `device` | `"cpu"` | `"cpu"` or `"cuda"` |
| `batch_search` | `False` | `True`: searches from parallel episodes share one encoder pass, about 4× faster; scores then vary by up to 0.0001 |

The deterministic baseline asks for `max(20, 5*K)` candidates. The paper agent forwards each
chosen `top_k` (the compact grammar permits 1–100); it is not silently reduced to 20 or to the
baseline's count. Its prompt-visible ledger holds 60 candidates. Paper-agent runs reject capped
or batched environments. The existing tool-noise setting remains available for controlled
corruption experiments; it does not change the frozen agent prompt.

## More

- **Internals,** design choices, and measured speed and memory: [docs/overview.md](docs/overview.md).
- **Install options and adapters:** [README.md](README.md).

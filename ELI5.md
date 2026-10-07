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
3. **It submits a report** of exactly K products, each with a short reason.
4. **The report is compared with the request's reference items:** K products taken from one real
   customer's purchases. The request was written to describe those products. K is between 3 and 7.

## The score

- **`sethit`:** the share of the reference items found among the first K submitted products, from
  0 to 1.
- **`valid`:** 1 if the report has exactly K different catalog products, each returned earlier by
  `search_products` or `get_complementary_products`. Otherwise 0.
- **`reward_direct`:** `sethit` if valid, otherwise −1. This is the RL reward.

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

- **`submit` says what is wrong** with a report (count, duplicates, unknown or unseen products)
  without revealing the reference items.
- **You can submit again.** The last report is the one scored.

## 4. Run a whole split

Every request in a split, with the built-in no-LLM baseline:

```python
from concurrent.futures import ThreadPoolExecutor
from recoatlas.baselines import run_baseline

n = len(env.splits["agent_hidden"])  # 1,960 requests
with ThreadPoolExecutor(32) as pool:
    scores = list(pool.map(lambda i: run_baseline(env.episode("agent_hidden", index=i)), range(n)))
print(sum(s["sethit"] for s in scores) / n)  # 0.0070
```

All threads share one `env`, so the tools are loaded once.

## 5. Plug in your own agent

For each request, your agent:

1. reads `ep.request` and `ep.k`;
2. calls tools with `ep.call(tool_name, **arguments)`. The tool descriptions, for a prompt or a
   tool schema, are in `recoatlas.env.TOOL_DOCS`;
3. finishes with `ep.submit(results)`.

Then average `ep.score()["sethit"]` over the split.

## 6. Train with RL

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
| `max_tool_results` | `20` | most products one call returns; `None` = no limit |
| `tool_noise` | `0.0` | share of complementary and near-duplicate results deliberately corrupted (by default, swapped for a random product of the same subcategory), to test robustness to faulty tools |
| `device` | `"cpu"` | `"cpu"` or `"cuda"` |
| `batch_search` | `False` | `True`: searches from parallel episodes share one encoder pass, about 4× faster; scores then vary by up to 0.0001 |

## More

- **Internals,** design choices, and measured speed and memory: [docs/overview.md](docs/overview.md).
- **Install options and adapters:** [README.md](README.md).

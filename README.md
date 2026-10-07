# recoatlas

RecoAtlas, a benchmark and RL environment for LLM recommendation agents, as a Python package. An
agent gets a shopping request and a number K, uses three tools on a fixed product catalog (search,
complementary products, near-duplicate removal), and submits K products. The report is scored
against the request's reference items, taken from one real customer's purchases. H&M only for now.

- **New here?** [ELI5.md](ELI5.md): what it is and how to use it, step by step.
- **How it works,** the design choices, and measured speed and memory:
  [docs/overview.md](docs/overview.md).
- **Working on the code** (people or coding agents): [AGENTS.md](AGENTS.md).

## Install

```bash
uv sync --extra tools          # the environment
uv sync --extra server         # + OpenEnv adapter and server
uv sync --extra trl            # + TRL adapter
uv sync --all-extras           # everything, for development
```

## Use

```python
import recoatlas

env = recoatlas.load("hm-clothing", device="cuda")  # first call downloads ~2.8 GB
ep = env.episode(split="agent_hidden", index=0)
ep.search_products("relaxed high-waisted shorts", top_k=10)
ep.submit([{"product_id": "...", "reasoning": "..."}, ...])
ep.score()  # {"sethit": ..., "valid": ..., "reward_direct": ...}
```

## Adapters

| module | for |
|---|---|
| `recoatlas.openenv_env` | OpenEnv environment; `recoatlas-openenv` serves many sessions from one process |
| `recoatlas.trl` | TRL `GRPOTrainer` `environment_factory` (see `examples/trl_grpo.py`) |
| `recoatlas.miles` | tool service for a Miles RecoAtlas agent loop (see `examples/miles_agent.py`) |

## Tests

```bash
uv run pytest -q            # fast tests
uv run pytest -q -m slow    # against the real H&M files
```

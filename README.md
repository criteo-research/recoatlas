# recoatlas

RecoAtlas, a benchmark and RL environment for LLM recommendation agents, as a Python package. An
agent gets a shopping request and a number K, uses three tools on a fixed product catalog (search,
complementary products, near-duplicate removal), and submits K products. The report is scored
against the request's reference items, taken from one real customer's purchases. Includes Amazon
Musical Instruments and H&M Clothing.

- **New here?** [ELI5.md](ELI5.md): what it is and how to use it, step by step.
- **How it works,** the design choices, and measured speed and memory:
  [docs/overview.md](docs/overview.md).
- **Working on the code** (people or coding agents): [AGENTS.md](AGENTS.md).
- **Rebuilding the frozen recommendation tools:**
  [unified-tool-creation/](unified-tool-creation/README.md).

The paper agent is a single [`Agent`](src/recoatlas/agent.py): the frozen optimized prompt,
all three tools, six decision rounds, advisory near-duplicate checking, and original report
filtering/backfilling. The RL adapters use the same report parser before computing rewards.
The separate [direct-relevance judge](src/recoatlas/judge.py) evaluates completed reports.
Local model profiles configure thinking on/off for Qwen, Gemma 4, Nemotron 3.5 Lightning and
GLM-4.7-Flash without changing the agent. See [model setup and validation](ELI5.md#choose-the-agent-model)
for serving requirements and the distinction between compatibility and paper-run reproduction.

For the four open models on both datasets, thinking on and off, use
[`scripts/run_open_models.sh`](scripts/run_open_models.sh). It starts local model servers,
checks development queries, and saves resumable results. See the
[overnight setup](ELI5.md#overnight-open-model-sweep) before running it on an allocated GPU host.

## Install

```bash
uv sync --extra tools          # the environment
uv sync --extra agent --extra grammar  # paper agent with compact vLLM JSON grammar
uv sync --extra llm            # judge only; no GPU/tool dependencies
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

For the paper-agent CLI, endpoint/API-key setup, and judging saved outputs, follow
[ELI5.md](ELI5.md#5-run-the-paper-agent). HF files are pinned and downloaded on first use;
model serving and a judge endpoint are separate dependencies. The `gpt-5.6-luna` name in the
paper is a deployment alias, not a guarantee that your API account has that model.

## Adapters

| module | for |
|---|---|
| `recoatlas.openenv_env` | OpenEnv environment; `recoatlas-openenv` serves many sessions from one process |
| `recoatlas.trl` | TRL `GRPOTrainer` `environment_factory` (see `examples/trl_grpo.py`) |
| `recoatlas.miles` | tool service for a Miles RecoAtlas agent loop (see `examples/miles_agent.py`) |

These adapters keep their RL interfaces. Native tool-calling training examples are integration
examples, not claims that their prompts or complete training trajectories match the paper agent.
The Miles example requires a checkout containing `examples/recoatlas/`, not the upstream repo alone.

## Tests

```bash
uv run pytest -q            # fast tests
uv run pytest -q -m slow    # against the real H&M files
```

## License

RecoAtlas code is released under the [Apache License 2.0](LICENSE). Downloaded datasets, catalog
content, base models, and derived tool artifacts remain subject to the terms identified in their
respective Hugging Face repositories; the code license does not relicense third-party material.

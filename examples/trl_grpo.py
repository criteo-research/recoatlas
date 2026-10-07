"""A few GRPO steps with TRL on H&M requests; RecoAtlas runs in the training process.

uv pip install "trl[vllm]" "recoatlas[trl]"
RECOATLAS_DEVICE=cuda python examples/trl_grpo.py --steps 3
"""

import argparse

from trl import GRPOConfig, GRPOTrainer

from recoatlas.trl import RecoAtlasEnvironment, trl_dataset

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
parser.add_argument("--steps", type=int, default=3)
parser.add_argument("--output-dir", default="trl-out")
args = parser.parse_args()

config = GRPOConfig(
    output_dir=args.output_dir,
    max_steps=args.steps,
    per_device_train_batch_size=8,
    num_generations=4,
    max_completion_length=2048,
    max_tool_calling_iterations=6,
    learning_rate=1e-6,
    bf16=True,
    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.3,
    vllm_max_model_length=8192,
    chat_template_kwargs={"enable_thinking": False},
    logging_steps=1,
    report_to="none",
    save_strategy="no",
)
trainer = GRPOTrainer(
    model=args.model,
    args=config,
    train_dataset=trl_dataset("agent_dev"),
    environment_factory=RecoAtlasEnvironment,
)
trainer.train()

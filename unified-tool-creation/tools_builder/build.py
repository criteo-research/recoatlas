"""Build artifacts for hybrid recommendation system."""

import argparse
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from dotenv import load_dotenv

load_dotenv()

from tools_builder.stage1_data_base import run_stage1
from tools_builder.stage2_query_gen import run_stage2
from tools_builder.stage3_train import run_stage3
from tools_builder.stage4_complementarity import run_stage4
from tools_builder.hf_upload import preflight_hf_repo
from tools_builder.storage import (
    check_ephemeral_budget,
    check_persistent_budget,
    cleanup_run,
    ensure_scratch_layout,
    run_dir,
    validate_bulk_paths_are_ephemeral,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="Musical_Instruments")
    parser.add_argument("--stage", type=str, default="all", choices=["1", "2", "3", "4", "all"])
    parser.add_argument("--train", action="store_true", default=False, help="Run training stages when --stage=all")
    parser.add_argument("--max_users", type=int, default=0, help="Max users to keep (0 = no limit)")
    parser.add_argument("--min_reviews_per_user", type=int, default=1)
    parser.add_argument("--min_reviews_per_item", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--query_batch_size", type=int, default=64)
    parser.add_argument(
        "--benchmark_split_manifest",
        type=str,
        default=None,
        help=(
            "Protected 70/15/15 user manifest. When set, retrieval and "
            "complementarity consume shared_train users only."
        ),
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default=os.getenv("RECOATLAS_QUERY_MODEL", "gpt-5.6-luna"),
        help="OpenAI-compatible model used for catalog query generation",
    )
    parser.add_argument(
        "--query_variants_per_product",
        type=int,
        default=20,
        help="Candidate query variants requested once per unique product",
    )
    parser.add_argument(
        "--reuse-stable-product-queries",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Seed split-bound retrieval generation from existing catalog-derived "
            "product queries, then generate only missing shared_train products."
        ),
    )
    parser.add_argument(
        "--product-query-seed-path",
        type=str,
        default=None,
        help="Optional local product_queries.json seed; useful for offline runs.",
    )
    parser.add_argument(
        "--query-cache-only",
        action="store_true",
        help=(
            "Never call the query-generation LLM. Products absent from the local "
            "or restored catalog-query seed receive no retrieval training rows."
        ),
    )
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="Gradient accumulation steps (effective batch = train_batch_size * grad_accum_steps)")
    parser.add_argument("--retrieval_lr", type=float, default=7.406576281631978e-5)
    parser.add_argument(
        "--retrieval_lr_schedule",
        choices=["linear", "constant"],
        default="constant",
        help="Retrieval learning-rate schedule",
    )
    parser.add_argument(
        "--retrieval_warmup_ratio",
        type=float,
        default=0.0,
        help="Fraction of optimizer steps used for LR warmup",
    )
    parser.add_argument("--contrastive_temperature", type=float, default=0.020090056606207288)
    parser.add_argument("--hard_negatives_per_query", type=int, default=4)
    parser.add_argument("--random_negatives_per_query", type=int, default=4)
    parser.add_argument("--hard_negative_candidates", type=int, default=100)
    parser.add_argument("--hard_negative_positive_ratio", type=float, default=0.95)
    parser.add_argument(
        "--retrieval_mining_run",
        type=str,
        default=None,
        help="Optional run ID whose compatible mined-negative cache seeds this run",
    )
    parser.add_argument("--retrieval_patience", type=int, default=2)
    parser.add_argument(
        "--publish-stable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Publish the selected retrieval model to the stable category path. "
            "Disable for isolated ablations; run-specific checkpoints are still uploaded."
        ),
    )
    parser.add_argument(
        "--allow-benchmark-stable-publish",
        action="store_true",
        help=(
            "Allow a split-bound benchmark training run to replace stable tool "
            "artifacts. Omit this during smoke tests and feasibility experiments."
        ),
    )
    parser.add_argument("--price_min", type=float, default=0.99)
    parser.add_argument("--price_max", type=float, default=10000.0)
    parser.add_argument("--min_year", type=int, default=None, help="Minimum product year (from 'Date First Available')")
    parser.add_argument("--max_year", type=int, default=None, help="Maximum product year (from 'Date First Available')")
    parser.add_argument(
        "--hf_repo",
        type=str,
        default="",
        help=(
            "Optional private model repository for resumable training state. "
            "Public inference artifacts are published separately with publish_tools.py."
        ),
    )
    parser.add_argument(
        "--hf_prefix",
        type=str,
        default=None,
        help="Optional folder inside the Hugging Face repository (defaults to category).",
    )
    parser.add_argument(
        "--copurchase_window_days",
        type=int,
        default=7,
        help="Maximum days between co-purchased products; use -1 for the full user history",
    )
    parser.add_argument("--backbone", type=str, default="qwen", choices=["qwen", "e5"],
                        help="Embedding backbone: qwen (Qwen3-0.6B, last-token pool) or e5 (e5-base-v2, mean pool)")
    parser.add_argument("--query_instruction", type=str, default="Given a shopping query, retrieve relevant products.")
    parser.add_argument("--query_max_length", type=int, default=256)
    parser.add_argument("--item_max_length", type=int, default=512)
    parser.add_argument("--min_pmi", type=float, default=0.0, help="Minimum PMI threshold for complementarity (0.0 = standard PPMI)")
    parser.add_argument("--comp_epochs", type=int, default=5, help="Epochs for complementarity training")
    parser.add_argument("--comp_batch_size", type=int, default=1024, help="Batch size for complementarity projection training")
    parser.add_argument("--comp_temperature", type=float, default=0.081621255477171, help="InfoNCE temperature for complementarity training")
    parser.add_argument("--comp_dropout", type=float, default=0.1, help="Dropout in complementarity projection heads")
    parser.add_argument("--comp_proj_dim", type=int, default=256, help="Complementarity projection dimension")
    parser.add_argument("--comp_lr", type=float, default=0.00040792119058488896)
    parser.add_argument("--comp_patience", type=int, default=2)
    parser.add_argument(
        "--comp_checkpoint_every",
        "--comp-checkpoint-every",
        dest="comp_checkpoint_every",
        type=int,
        default=0,
        help=(
            "Retain a numbered complementarity checkpoint every N epochs in "
            "addition to latest/best (0 keeps only rotating checkpoints)"
        ),
    )
    parser.add_argument("--comp_min_pair_count", type=int, default=1)
    parser.add_argument("--comp_validation_user_ratio", type=float, default=0.1)
    parser.add_argument("--comp_positives_per_anchor", type=int, default=8)
    parser.add_argument("--comp_hard_negatives", type=int, default=0)
    parser.add_argument("--comp_random_negatives", type=int, default=8)
    parser.add_argument("--comp_hard_negative_candidates", type=int, default=100)
    parser.add_argument("--comp_hard_negative_margin", type=float, default=0.02)
    parser.add_argument(
        "--comp-cross-subcategory",
        dest="comp_cross_subcategory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train and evaluate only cross-subcategory complement relationships",
    )
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None,
                        help="LoRA target modules (default: auto-resolved from model)")
    parser.add_argument("--early_stop_metric", type=str, default="val_multi_positive_mrr",
                        help="Metric for early stopping checkpoint selection")
    parser.add_argument(
        "--allow_cpu",
        action="store_true",
        help="Allow embedding/training stages to run on CPU (normally refused as impractically slow)",
    )
    parser.add_argument("--run_id", type=str, default=None,
                        help="Scratch run identifier (default: timestamped unique ID)")
    parser.add_argument("--resume", action="store_true",
                        help="Restore resumable query/model checkpoints from Hugging Face")
    parser.add_argument(
        "--cleanup",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Remove the local run after verified uploads (default: yes for --stage=all)",
    )
    parser.add_argument(
        "--retain-release-artifacts",
        action="store_true",
        help=(
            "Keep the selected inference payloads locally so scripts/publish_tools.py "
            "can publish the clean five-file release. Transient state is still pruned."
        ),
    )
    args = parser.parse_args()

    if args.query_variants_per_product <= 0:
        parser.error("--query_variants_per_product must be positive")
    if args.epochs <= 0 or args.comp_epochs <= 0:
        parser.error("--epochs and --comp_epochs must be positive")
    if args.contrastive_temperature <= 0:
        parser.error("--contrastive_temperature must be positive")
    if args.retrieval_lr <= 0:
        parser.error("--retrieval_lr must be positive")
    if not 0 <= args.retrieval_warmup_ratio < 1:
        parser.error("--retrieval_warmup_ratio must be in [0, 1)")
    if args.lora_r <= 0 or args.lora_alpha <= 0:
        parser.error("--lora_r and --lora_alpha must be positive")
    if not 0 <= args.lora_dropout < 1:
        parser.error("--lora_dropout must be in [0, 1)")
    if not 0 < args.hard_negative_positive_ratio < 1:
        parser.error("--hard_negative_positive_ratio must be between zero and one")
    if args.retrieval_patience <= 0:
        parser.error("--retrieval_patience must be positive")
    if args.comp_lr <= 0:
        parser.error("--comp_lr must be positive")
    if args.comp_proj_dim <= 0:
        parser.error("--comp_proj_dim must be positive")
    if args.comp_patience <= 0 or args.comp_min_pair_count <= 0:
        parser.error("--comp_patience and --comp_min_pair_count must be positive")
    if args.comp_checkpoint_every < 0:
        parser.error("--comp_checkpoint_every cannot be negative")
    if not 0 < args.comp_validation_user_ratio < 1:
        parser.error("--comp_validation_user_ratio must be between zero and one")
    if args.comp_positives_per_anchor <= 0:
        parser.error("--comp_positives_per_anchor must be positive")
    if args.comp_hard_negatives < 0 or args.comp_random_negatives < 0:
        parser.error("complementarity negative counts cannot be negative")
    if args.comp_hard_negatives + args.comp_random_negatives <= 0:
        parser.error("Complementarity training requires at least one negative")
    if args.comp_hard_negative_candidates < args.comp_hard_negatives:
        parser.error("--comp_hard_negative_candidates must cover --comp_hard_negatives")
    if args.resume and not (args.run_id or os.getenv("RECO_RUN_ID")):
        parser.error("--resume requires --run_id (or RECO_RUN_ID) to identify remote state")

    ensure_scratch_layout()
    validate_bulk_paths_are_ephemeral()
    check_persistent_budget()
    check_ephemeral_budget(reason="build startup")
    preflight_hf_repo(args.hf_repo)

    resolved_run_dir = run_dir(args.category, args.run_id)
    if resolved_run_dir.exists() and any(resolved_run_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Run directory already contains artifacts: {resolved_run_dir}. "
            "Use --resume with the same --run_id or choose a new run ID."
        )
    save_dir = str(resolved_run_dir)
    os.makedirs(save_dir, exist_ok=True)
    args.run_id = os.path.basename(save_dir)
    print(f"Working directory: {save_dir}")

    accelerator_stages = {"1", "3", "4", "all"}
    if args.stage in accelerator_stages and not args.allow_cpu:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available, so refusing an impractically slow CPU embedding/training run. "
                "Attach a working GPU or pass --allow_cpu explicitly."
            )

    def run_checked(stage_name, function):
        check_persistent_budget(reason=f"stage {stage_name}")
        check_ephemeral_budget(reason=f"stage {stage_name} startup")
        function(args, save_dir)
        check_ephemeral_budget(reason=f"stage {stage_name} completion")

    def maybe_cleanup(default):
        cleanup = args.cleanup if args.cleanup is not None else default
        if not cleanup:
            return
        if not args.hf_repo:
            raise ValueError("Refusing --cleanup without --hf_repo; no durable copy exists.")
        cleanup_run(save_dir)
        print(f"Removed verified local run: {save_dir}")

    if args.stage == "1":
        run_checked("1", run_stage1)
        maybe_cleanup(False)
        return

    if args.stage == "2":
        run_checked("2", run_stage2)
        maybe_cleanup(False)
        return

    if args.stage == "3":
        run_checked("3", run_stage3)
        maybe_cleanup(False)
        return

    if args.stage == "4":
        run_checked("4", run_stage4)
        maybe_cleanup(False)
        return

    # args.stage == "all"
    run_checked("1", run_stage1)
    if args.train:
        run_checked("2", run_stage2)
        run_checked("3", run_stage3)
        run_checked("4", run_stage4)
    else:
        print("Skipping stages 2-4 (--train not set).")

    # A full build must retain its five selected payloads for the clean release
    # publisher. Intermediate scratch can be removed explicitly after publication.
    maybe_cleanup(False)


if __name__ == "__main__":
    main()

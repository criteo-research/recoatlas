"""Stage 2: query generation and product-disjoint split."""

import json
import os

import pandas as pd

from data import SimpleQueryGenerator
from tools_builder.benchmark_split import (
    filter_interactions_to_split,
    load_user_split_manifest,
    public_split_metadata,
)
from tools_builder.hf_upload import (
    hf_category,
    hf_run,
    prune_verified_files,
    restore_files,
    upload_files,
)
from tools_builder.recsys_model import split_queries_by_product
from tools_builder.storage import atomic_json_dump, check_ephemeral_budget


def publish_stage2(args, save_dir: str):
    if not args.hf_repo:
        return
    remote_category = (
        hf_run(args)
        if getattr(args, "benchmark_split_manifest", None)
        else hf_category(args)
    )
    files = [
        "queries/queries.json",
        "queries/query_types.json",
        "queries/queries_train.json",
        "queries/queries_test.json",
        "queries/product_queries.json",
        "queries/product_queries_train.json",
        "queries/product_queries_test.json",
    ]
    if getattr(args, "benchmark_split_manifest", None):
        files.append("queries/benchmark_split_metadata.json")
    upload_files(args.hf_repo, remote_category, save_dir, files)


def run_stage2(args, save_dir: str):
    data_dir = os.path.join(save_dir, "data")
    queries_dir = os.path.join(save_dir, "queries")
    os.makedirs(queries_dir, exist_ok=True)

    stable_query_files = [
        "queries/queries.json",
        "queries/query_types.json",
        "queries/queries_train.json",
        "queries/queries_test.json",
        "queries/product_queries.json",
        "queries/product_queries_train.json",
        "queries/product_queries_test.json",
    ]
    split_manifest_path = getattr(args, "benchmark_split_manifest", None)
    if split_manifest_path:
        stable_query_files.append("queries/benchmark_split_metadata.json")
    if getattr(args, "resume", False):
        restore_files(
            args.hf_repo,
            hf_run(args) if split_manifest_path else hf_category(args),
            save_dir,
            stable_query_files,
        )
        if all(os.path.exists(os.path.join(save_dir, path)) for path in stable_query_files):
            if split_manifest_path:
                split_manifest = load_user_split_manifest(split_manifest_path)
                with open(
                    os.path.join(save_dir, "queries/benchmark_split_metadata.json")
                ) as handle:
                    restored_metadata = json.load(handle)
                if restored_metadata.get("fingerprint") != split_manifest["fingerprint"]:
                    raise ValueError(
                        "Restored retrieval queries use a different benchmark split"
                    )
            print("Stage 2 already complete; restored balanced query artifacts.")
            return

    restore_files(
        args.hf_repo,
        hf_category(args),
        save_dir,
        ["data/interactions.parquet", "data/subcategories.json"],
        required=True,
    )
    query_cache_filename = SimpleQueryGenerator.cache_filename_for_model(args.llm_model)
    if getattr(args, "resume", False):
        restore_files(
            args.hf_repo,
            hf_run(args),
            save_dir,
            [query_cache_filename],
        )

    interactions_path = f"{data_dir}/interactions.parquet"
    subcategories_path = f"{data_dir}/subcategories.json"
    missing = [
        path
        for path in (interactions_path, subcategories_path)
        if not os.path.exists(path)
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing required stage-1 artifacts: {missing}\n"
            f"Please run stage 1 first: python -m tools_builder.build --stage 1 --category {args.category}"
        )

    print(f"Loading interactions from {interactions_path}...")
    df = pd.read_parquet(interactions_path)
    split_manifest = None
    if split_manifest_path:
        split_manifest = load_user_split_manifest(split_manifest_path)
        if split_manifest.get("category") not in (None, args.category):
            raise ValueError(
                "Benchmark split category does not match this build: "
                f"{split_manifest.get('category')} != {args.category}"
            )
        before = df["user_id"].astype(str).nunique()
        df = filter_interactions_to_split(df, split_manifest, "shared_train")
        after = df["user_id"].astype(str).nunique()
        if not after:
            raise ValueError("Benchmark shared_train contains no stage-1 interactions")
        print(
            "Applied protected benchmark split before query generation: "
            f"{after:,}/{before:,} interaction users retained (shared_train only)."
        )
    with open(subcategories_path) as handle:
        subcategories = json.load(handle)

    stable_product_query_seed = {}
    local_seed_path = getattr(args, "product_query_seed_path", None)
    if local_seed_path:
        with open(local_seed_path) as handle:
            candidate_seed = json.load(handle)
        if not isinstance(candidate_seed, dict):
            raise ValueError("Local product-query seed must be a JSON object")
        stable_product_query_seed.update(candidate_seed)
        print(
            "Loaded local product-query seed with "
            f"{len(candidate_seed):,} catalog entries."
        )
    if (
        split_manifest_path
        and getattr(args, "reuse_stable_product_queries", True)
        and not local_seed_path
    ):
        restore_files(
            args.hf_repo,
            hf_category(args),
            save_dir,
            ["queries/product_queries.json"],
        )
        stable_seed_path = os.path.join(queries_dir, "product_queries.json")
        if os.path.exists(stable_seed_path):
            with open(stable_seed_path) as handle:
                candidate_seed = json.load(handle)
            if isinstance(candidate_seed, dict):
                stable_product_query_seed = candidate_seed
            print(
                "Seeded benchmark-bound retrieval generation with "
                f"{len(stable_product_query_seed):,} existing catalog query entries."
            )

    print("Initializing simple product-query generator...")
    mc = 20  # maximum concurrent OpenAI-compatible requests
    generator = SimpleQueryGenerator(
        item_to_subcategory=subcategories["item_to_subcategory"],
        cache_dir=save_dir,
        model=args.llm_model,
        num_variants=args.query_variants_per_product,
        max_concurrent=mc,
        cache_only=getattr(args, "query_cache_only", False),
    )
    # These queries were derived only from public catalog text. Reusing them does
    # not expose dev/hidden behavior; filtering to shared_train products happens
    # below before training artifacts are written.
    generator.cache.update(stable_product_query_seed)

    def publish_snapshot(path):
        if not args.hf_repo:
            return
        relative = os.path.relpath(path, save_dir)
        print(f"Publishing resumable query cache: {relative}")
        upload_files(args.hf_repo, hf_run(args), save_dir, [relative])

    generator.snapshot_callback = publish_snapshot

    pair_count = len(
        df[df["rating"] >= 4.0][["user_id", "parent_asin"]].drop_duplicates()
    )
    check_ephemeral_budget(pair_count * 2048, reason="query artifacts")
    queries = generator.generate_all(
        df, pos_threshold=4.0, batch_size=args.query_batch_size
    )
    product_queries = generator.product_variants({item_id for _, item_id in queries})
    eligible_products = set(
        df.loc[df["rating"] >= 4.0, "parent_asin"].dropna().astype(str)
    )
    missing_query_products = eligible_products - set(product_queries)
    if missing_query_products:
        policy = "excluded without LLM calls" if getattr(args, "query_cache_only", False) else "missing after generation"
        print(
            f"Catalog-query coverage: {len(product_queries):,}/{len(eligible_products):,}; "
            f"{len(missing_query_products):,} products {policy}."
        )
    generator.unload()
    query_types = {key: "simple" for key in queries}

    atomic_json_dump({f"{k[0]}||{k[1]}": v for k, v in queries.items()}, f"{queries_dir}/queries.json")
    atomic_json_dump(
        {f"{k[0]}||{k[1]}": query_types[k] for k in queries.keys()},
        f"{queries_dir}/query_types.json",
    )

    print("\nSplitting queries by product...")
    train_queries, val_queries = split_queries_by_product(queries, val_ratio=0.1)
    train_products = {item_id for _, item_id in train_queries}
    val_products = {item_id for _, item_id in val_queries}
    product_queries_train = {
        item_id: variants for item_id, variants in product_queries.items()
        if item_id in train_products
    }
    product_queries_test = {
        item_id: variants for item_id, variants in product_queries.items()
        if item_id in val_products
    }
    print(
        f"Product split: {len(train_products):,} train, "
        f"{len(val_products):,} validation (overlap: "
        f"{len(train_products & val_products)})"
    )

    atomic_json_dump(
        {f"{k[0]}||{k[1]}": v for k, v in train_queries.items()},
        f"{queries_dir}/queries_train.json",
    )
    atomic_json_dump(
        {f"{k[0]}||{k[1]}": v for k, v in val_queries.items()},
        f"{queries_dir}/queries_test.json",
    )
    atomic_json_dump(product_queries, f"{queries_dir}/product_queries.json")
    atomic_json_dump(
        product_queries_train, f"{queries_dir}/product_queries_train.json"
    )
    atomic_json_dump(
        product_queries_test, f"{queries_dir}/product_queries_test.json"
    )
    if split_manifest is not None:
        metadata = public_split_metadata(split_manifest)
        metadata.update(
            {
                "source_partition": "shared_train",
                "interaction_users_retained": after,
                "retrieval_train_products": len(train_products),
                "retrieval_selection_products": len(val_products),
                "retrieval_selection_policy": "product_disjoint_within_shared_train",
                "stable_product_query_seed_count": len(stable_product_query_seed),
                "query_cache_only": bool(getattr(args, "query_cache_only", False)),
                "products_without_query_supervision": len(missing_query_products),
            }
        )
        atomic_json_dump(
            metadata, f"{queries_dir}/benchmark_split_metadata.json", indent=2
        )
    print(f"Saved {len(train_queries):,} train queries, {len(val_queries):,} test queries")
    print(
        "Saved balanced product variants: "
        f"{sum(map(len, product_queries_train.values())):,} train, "
        f"{sum(map(len, product_queries_test.values())):,} validation"
    )

    publish_stage2(args, save_dir)
    if args.hf_repo:
        prune_verified_files(save_dir, [query_cache_filename])

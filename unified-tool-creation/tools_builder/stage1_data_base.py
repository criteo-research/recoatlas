"""Stage 1: data preparation + base embeddings artifacts."""

import gc
import json
import os

import faiss
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from data import ProductCatalog
from tools_builder.hf_upload import hf_category, restore_files, upload_files
from tools_builder.recsys_model import BACKBONE_CONFIGS
from tools_builder.utils import (
    compute_copurchase,
    compute_pmi,
    compute_popularity,
    compute_subcategory_mapping,
    encode_texts_base,
    filter_copurchase,
    load_data,
    merge_and_filter,
    save_config,
)
from tools_builder.storage import atomic_json_dump, atomic_torch_save, check_ephemeral_budget


def build_faiss_index(
    embeddings: torch.Tensor,
    hnsw_m: int = 32,
    hnsw_ef_construction: int = 200,
    hnsw_ef_search: int = 128,
) -> faiss.Index:
    """Build HNSW inner-product index on normalized vectors."""
    embs_np = embeddings.cpu().numpy().astype("float32")
    dim = embs_np.shape[1]
    index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = hnsw_ef_construction
    index.hnsw.efSearch = hnsw_ef_search
    index.add(embs_np)
    return index

def stage1_artifact_files(backbone: str) -> list[str]:
    return [
        "data/product_texts.json",
        "data/item_order.json",
        "data/copurchase.json",
        "data/copurchase_pmi.json",
        "data/popularity.json",
        "data/subcategories.json",
        f"data/config_{backbone}.json",
        "data/interactions.parquet",
        f"embeddings/item_embeddings_{backbone}.pt",
        f"embeddings/item_index_{backbone}.faiss",
    ]


def publish_stage1(args, save_dir: str):
    if not args.hf_repo:
        return
    upload_files(
        args.hf_repo,
        hf_category(args),
        save_dir,
        stage1_artifact_files(args.backbone),
    )


def publish_stage1_files(args, save_dir: str, files: list[str]):
    if args.hf_repo:
        upload_files(args.hf_repo, hf_category(args), save_dir, files)


def run_stage1(args, save_dir: str):
    data_dir = os.path.join(save_dir, "data")
    embeddings_dir = os.path.join(save_dir, "embeddings")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)
    backbone = args.backbone

    if getattr(args, "resume", False):
        files = stage1_artifact_files(backbone)
        restore_files(
            args.hf_repo,
            hf_category(args),
            save_dir,
            files,
            required=False,
        )
        missing = [name for name in files if not os.path.isfile(os.path.join(save_dir, name))]
        if not missing:
            print("Stage 1 already complete; restored and verified durable artifacts.")
            return
        print(
            "Stage 1 is incomplete locally and on Hugging Face; building missing "
            f"fresh-run artifacts ({len(missing)} remaining)."
        )

    print(f"Loading data for {args.category}...")
    df_reviews, df_meta = load_data(args.category)

    review_cols = ["user_id", "parent_asin", "timestamp", "rating", "title", "text"]
    df_reviews = df_reviews[[c for c in review_cols if c in df_reviews.columns]]
    gc.collect()

    print("Computing co-purchase matrix on raw data...")
    copurchase_cols = ["user_id", "parent_asin", "rating", "timestamp"]
    df_copurchase = df_reviews[copurchase_cols].copy()
    df_copurchase["timestamp"] = pd.to_datetime(df_copurchase["timestamp"], unit="ms")
    window = args.copurchase_window_days if args.copurchase_window_days >= 0 else None
    copurchase_raw = compute_copurchase(
        df_copurchase, min_rating=4.0, min_count=1, time_window_days=window
    )
    del df_copurchase
    gc.collect()

    year_range = (args.min_year, args.max_year) if args.min_year or args.max_year else None
    df = merge_and_filter(
        df_reviews,
        df_meta,
        min_reviews_per_user=args.min_reviews_per_user,
        min_reviews_per_item=args.min_reviews_per_item,
        max_users=args.max_users,
        price_range=(args.price_min, args.price_max),
        year_range=year_range,
    )
    del df_reviews
    gc.collect()

    catalog_items = set(df["parent_asin"].unique())

    print("Filtering co-purchase to catalog items...")
    copurchase = filter_copurchase(copurchase_raw, catalog_items)

    min_pmi = getattr(args, "min_pmi", 0.0)
    print(f"Computing PPMI from co-purchase (min_pmi={min_pmi})...")
    copurchase_pmi = compute_pmi(copurchase, min_pmi=min_pmi)
    print(f"PMI graph: {len(copurchase_pmi)} items, "
          f"{sum(len(v) for v in copurchase_pmi.values())} edges")

    print("Generating product representations...")
    catalog = ProductCatalog(df)
    product_reps = catalog.generate_all()

    dim = 1024 if backbone == "qwen" else 768
    item_count = len(product_reps)
    edge_count = sum(len(neighbors) for neighbors in copurchase.values())
    dataframe_bytes = int(df.memory_usage(index=True, deep=True).sum())
    # Parquet + JSON/text + base embeddings + HNSW + graph serialization.
    expected_outputs = (
        dataframe_bytes
        + item_count * 4096
        + item_count * dim * 4 * 3
        + edge_count * 128
    )
    check_ephemeral_budget(expected_outputs, reason="stage-1 artifacts")

    atomic_json_dump(product_reps, f"{data_dir}/product_texts.json")

    item_order = list(product_reps.keys())
    atomic_json_dump(item_order, f"{data_dir}/item_order.json")

    atomic_json_dump(copurchase, f"{data_dir}/copurchase.json")

    atomic_json_dump(copurchase_pmi, f"{data_dir}/copurchase_pmi.json")

    print("Computing popularity by subcategory...")
    popularity = compute_popularity(df_meta, catalog_items, product_reps)
    atomic_json_dump(popularity, f"{data_dir}/popularity.json")

    print("Computing subcategory mapping...")
    subcategories = compute_subcategory_mapping(df_meta, catalog_items)
    atomic_json_dump(subcategories, f"{data_dir}/subcategories.json")
    del df_meta
    gc.collect()

    save_config(args, data_dir, backbone)

    interactions_cols = [
        "user_id",
        "parent_asin",
        "rating",
        "timestamp",
        "product_title",
        "product_description",
        "product_price",
    ]
    available_cols = [c for c in interactions_cols if c in df.columns]
    interactions_path = f"{data_dir}/interactions.parquet"
    interactions_tmp = f"{interactions_path}.tmp"
    df[available_cols].to_parquet(interactions_tmp, index=False)
    os.replace(interactions_tmp, interactions_path)

    publish_stage1_files(args, save_dir, [
        "data/product_texts.json",
        "data/item_order.json",
        "data/copurchase.json",
        "data/copurchase_pmi.json",
        "data/popularity.json",
        "data/subcategories.json",
        f"data/config_{backbone}.json",
        "data/interactions.parquet",
    ])

    item_texts = list(product_reps.values())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bcfg = BACKBONE_CONFIGS[backbone]
    embedding_model_name = bcfg["model_name"]
    check_ephemeral_budget(2 * 1024 ** 3, reason=f"cache embedding model {embedding_model_name}")
    print(f"Computing base embeddings with {embedding_model_name} on {device}...")
    source_kwargs = {"revision": bcfg["revision"]} if bcfg.get("revision") else {}
    tokenizer = AutoTokenizer.from_pretrained(
        embedding_model_name, padding_side=bcfg["pad_side"], **source_kwargs
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    encoder = AutoModel.from_pretrained(
        embedding_model_name, **source_kwargs
    ).eval().to(device)

    item_embs = encode_texts_base(
        item_texts,
        tokenizer,
        encoder,
        batch_size=32,
        max_length=args.item_max_length,
        prefix=bcfg["item_prefix"],
        normalize=True,
        pooling=bcfg["pooling"],
        eod_token=bcfg["eod_token"],
    )
    item_embs = item_embs.to(torch.float32)
    embedding_path = f"{embeddings_dir}/item_embeddings_{backbone}.pt"
    atomic_torch_save(
        {"item_embs": item_embs, "item_ids": item_order},
        embedding_path,
        expected_bytes=item_embs.numel() * item_embs.element_size(),
    )
    publish_stage1_files(args, save_dir, [f"embeddings/item_embeddings_{backbone}.pt"])
    del encoder, tokenizer

    print("Building FAISS index for base embeddings...")
    base_index = build_faiss_index(item_embs)
    index_path = f"{embeddings_dir}/item_index_{backbone}.faiss"
    index_tmp = f"{index_path}.tmp"
    faiss.write_index(base_index, index_tmp)
    os.replace(index_tmp, index_path)
    check_ephemeral_budget(reason="completed FAISS index")
    publish_stage1_files(args, save_dir, [f"embeddings/item_index_{backbone}.faiss"])

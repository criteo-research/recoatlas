"""H&M catalog and purchase adapter for the shared RecoAtlas trainers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tools_builder.benchmark_split import load_user_split_manifest
from tools_builder.storage import atomic_json_dump, check_ephemeral_budget


ARTICLE_FIELDS = {
    "article_id",
    "prod_name",
    "detail_desc",
    "product_type_name",
    "colour_group_name",
    "graphical_appearance_name",
    "index_name",
    "section_name",
}
TRANSACTION_FIELDS = {"customer_id", "article_id", "t_dat"}


def _article_ids(values: pd.Series) -> pd.Series:
    """Preserve H&M identifiers as ten-character zero-padded strings."""
    numeric = pd.to_numeric(values, errors="raise").astype("int64")
    return numeric.astype(str).str.zfill(10)


def _read_table(path: Path, **kwargs) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, **kwargs)
    if path.suffix.lower() in {".csv", ".gz"}:
        return pd.read_csv(path, **kwargs)
    raise ValueError(f"Unsupported input table: {path}")


def _transaction_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.suffix.lower() in {".csv", ".parquet", ".gz"}
    )
    if not files:
        raise FileNotFoundError(f"No transaction tables found below {path}")
    return files


def _transaction_frames(path: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
    for source in _transaction_files(path):
        if source.suffix.lower() == ".parquet":
            parquet = pq.ParquetFile(source)
            for batch in parquet.iter_batches(batch_size=chunk_size):
                yield batch.to_pandas()
        else:
            yield from pd.read_csv(source, chunksize=chunk_size)


def canonical_catalog(articles: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Apply the paper catalog filters and construct the frozen item text."""
    missing = ARTICLE_FIELDS - set(articles.columns)
    if missing:
        raise ValueError(f"H&M articles are missing columns: {sorted(missing)}")
    frame = articles.copy()
    frame["item_id"] = _article_ids(frame["article_id"])
    frame["title"] = frame["prod_name"].fillna("").astype(str).str.strip()
    frame["description"] = (
        frame["detail_desc"]
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    frame["subcategory"] = (
        frame["product_type_name"].fillna("").astype(str).str.strip()
    )
    known_type = ~frame["subcategory"].str.casefold().isin({"", "unknown", "nan"})
    eligible = (
        frame["title"].str.len().gt(0)
        & frame["description"].str.len().ge(15)
        & known_type
    )
    frame = frame.loc[eligible].copy()
    if frame["item_id"].duplicated().any():
        raise ValueError("H&M article IDs are not unique")

    texts = {}
    subcategories = {}
    for row in frame.itertuples(index=False):
        texts[row.item_id] = (
            f"Title: {row.title[:200]} | Type: {row.subcategory}"
            f" | Color: {row.colour_group_name}"
            f" | Pattern: {row.graphical_appearance_name}"
            f" | Department: {row.index_name}"
            f" | Section: {row.section_name}"
            f" | Description: {row.description[:500]}"
        )
        subcategories[row.item_id] = row.subcategory
    return frame, texts, subcategories


def prepare_hm_inputs(
    articles_path: str | Path,
    transactions_path: str | Path,
    split_manifest_path: str | Path,
    save_dir: str | Path,
    *,
    chunk_size: int = 250_000,
) -> dict:
    """Create common stage artifacts from H&M articles and purchases.

    The searchable catalog contains every eligible article. Behavioral files
    contain only ``shared_train`` users. ``rating=5`` is a compatibility value
    denoting an implicit positive, not an observed source rating.
    """
    save_dir = Path(save_dir)
    data_dir = save_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_user_split_manifest(split_manifest_path)
    if manifest.get("category") not in (None, "H_and_M"):
        raise ValueError("The protected split is not an H&M split")
    allowed_users = set(map(str, manifest["splits"]["shared_train"]))

    articles = _read_table(Path(articles_path))
    catalog, texts, item_to_subcategory = canonical_catalog(articles)
    del articles
    eligible_items = set(texts)
    metadata = catalog.set_index("item_id")[['title', 'description']]

    atomic_json_dump(texts, data_dir / "product_texts.json")
    atomic_json_dump(list(texts), data_dir / "item_order.json")
    atomic_json_dump(
        {
            "item_to_subcategory": item_to_subcategory,
            "subcategory_to_items": {
                category: [item for item, value in item_to_subcategory.items() if value == category]
                for category in sorted(set(item_to_subcategory.values()))
            },
            "subcategories": sorted(set(item_to_subcategory.values())),
        },
        data_dir / "subcategories.json",
    )
    catalog.to_parquet(data_dir / "catalog.parquet", index=False)

    interactions_path = data_dir / "interactions.parquet"
    temporary = interactions_path.with_suffix(".parquet.tmp")
    schema = pa.schema(
        [
            ("user_id", pa.string()),
            ("parent_asin", pa.string()),
            ("rating", pa.float32()),
            ("timestamp", pa.timestamp("ns")),
            ("product_title", pa.string()),
            ("product_description", pa.string()),
            ("product_price", pa.float64()),
        ]
    )
    scanned = retained = 0
    with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
        for frame in _transaction_frames(Path(transactions_path), chunk_size):
            missing = TRANSACTION_FIELDS - set(frame.columns)
            if missing:
                raise ValueError(
                    f"H&M transactions are missing columns: {sorted(missing)}"
                )
            scanned += len(frame)
            users = frame["customer_id"].astype(str)
            items = _article_ids(frame["article_id"])
            keep = users.isin(allowed_users) & items.isin(eligible_items)
            if not keep.any():
                continue
            selected_items = items[keep]
            output = pd.DataFrame(
                {
                    "user_id": users[keep].to_numpy(),
                    "parent_asin": selected_items.to_numpy(),
                    "rating": 5.0,
                    "timestamp": pd.to_datetime(frame.loc[keep, "t_dat"], errors="raise").to_numpy(),
                    "product_title": metadata.loc[selected_items, "title"].to_numpy(),
                    "product_description": metadata.loc[selected_items, "description"].to_numpy(),
                    "product_price": pd.to_numeric(
                        frame.loc[keep, "price"], errors="coerce"
                    ).to_numpy()
                    if "price" in frame
                    else float("nan"),
                }
            )
            retained += len(output)
            writer.write_table(pa.Table.from_pandas(output, schema=schema, preserve_index=False))
            check_ephemeral_budget(reason="H&M interaction preparation", force_scan=False)
    temporary.replace(interactions_path)
    if retained == 0:
        raise ValueError("No H&M shared_train purchases survived preprocessing")

    summary = {
        "category": "H_and_M",
        "catalog_items": len(texts),
        "source_purchases_scanned": scanned,
        "shared_train_purchases": retained,
        "shared_train_users": len(allowed_users),
        "split_fingerprint": manifest["fingerprint"],
        "positive_policy": "Every purchase is positive; rating=5 is a compatibility encoding.",
        "article_id_policy": "ten-character zero-padded string",
    }
    atomic_json_dump(summary, data_dir / "preparation.json", indent=2)
    return summary


def encode_hm_catalog(
    save_dir: str | Path,
    *,
    backbone: str = "qwen",
    batch_size: int = 32,
    max_length: int = 512,
) -> dict:
    """Encode the complete H&M catalog and build the shared FAISS index."""
    import faiss
    import torch
    from transformers import AutoModel, AutoTokenizer

    from tools_builder.recsys_model import BACKBONE_CONFIGS
    from tools_builder.stage1_data_base import build_faiss_index
    from tools_builder.storage import atomic_torch_save
    from tools_builder.utils import encode_texts_base

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to encode the H&M catalog")
    save_dir = Path(save_dir)
    embedding_dir = save_dir / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    texts = json.loads((save_dir / "data/product_texts.json").read_text())
    config = BACKBONE_CONFIGS[backbone]
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        padding_side=config["pad_side"],
        **({"revision": config["revision"]} if config.get("revision") else {}),
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        config["model_name"],
        **({"revision": config["revision"]} if config.get("revision") else {}),
    ).eval().cuda()
    embeddings = encode_texts_base(
        list(texts.values()),
        tokenizer,
        model,
        batch_size=batch_size,
        max_length=max_length,
        prefix=config["item_prefix"],
        normalize=True,
        pooling=config["pooling"],
        eod_token=config["eod_token"],
    ).float()
    payload = {"item_ids": list(texts), "item_embs": embeddings}
    output = embedding_dir / f"item_embeddings_{backbone}.pt"
    atomic_torch_save(payload, output, expected_bytes=embeddings.numel() * 4)
    index = build_faiss_index(embeddings)
    faiss.write_index(index, str(embedding_dir / f"item_index_{backbone}.faiss"))
    return {"catalog_items": len(texts), "embedding_dim": embeddings.shape[1]}

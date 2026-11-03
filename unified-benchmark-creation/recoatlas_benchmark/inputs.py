"""Adapters from the released Amazon and H&M source tables."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .mining import Candidate, mine_hm_encoded
from .splits import SPLIT_NAMES, validate_manifest


def read_table(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.casefold() == ".parquet":
        return pd.read_parquet(path, **kwargs)
    if path.suffix.casefold() in {".csv", ".gz"}:
        return pd.read_csv(path, **kwargs)
    raise ValueError(f"Unsupported table: {path}")


def input_files(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    files = sorted(
        value for value in path.iterdir() if value.suffix.casefold() in {".csv", ".gz", ".parquet"}
    )
    if not files:
        raise FileNotFoundError(f"No CSV or Parquet files below {path}")
    return files


def transaction_frames(path: str | Path, chunk_size: int = 250_000) -> Iterator[pd.DataFrame]:
    for source in input_files(path):
        if source.suffix.casefold() == ".parquet":
            for batch in pq.ParquetFile(source).iter_batches(batch_size=chunk_size):
                yield batch.to_pandas()
        else:
            yield from pd.read_csv(source, chunksize=chunk_size)


def load_amazon_data(data_dir: str | Path) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    """Load canonical artifacts produced by unified-tool-creation stage 1."""
    data_dir = Path(data_dir)
    interactions = pd.read_parquet(data_dir / "interactions.parquet")
    product_texts = json.loads((data_dir / "product_texts.json").read_text(encoding="utf-8"))
    subcategory_payload = json.loads((data_dir / "subcategories.json").read_text(encoding="utf-8"))
    subcategories = subcategory_payload.get("item_to_subcategory", subcategory_payload)
    if not isinstance(product_texts, dict) or not isinstance(subcategories, dict):
        raise TypeError("Invalid canonical Amazon catalog")
    return interactions, product_texts, subcategories


def canonical_hm_catalog(
    articles_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    articles = read_table(articles_path)
    required = {
        "article_id",
        "product_code",
        "prod_name",
        "detail_desc",
        "product_type_name",
        "colour_group_name",
        "graphical_appearance_name",
        "index_group_name",
        "index_name",
        "section_name",
    }
    if missing := required - set(articles):
        raise ValueError(f"H&M articles are missing columns: {sorted(missing)}")
    frame = articles.copy()
    frame["item_id"] = (
        pd.to_numeric(frame["article_id"], errors="raise").astype("int64").astype(str).str.zfill(10)
    )
    frame["title"] = frame["prod_name"].fillna("").astype(str).str.strip()
    frame["description"] = (
        frame["detail_desc"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    )
    frame["product_type"] = frame["product_type_name"].fillna("").astype(str).str.strip()
    known = ~frame["product_type"].str.casefold().isin({"", "unknown", "nan"})
    frame = frame[
        frame["title"].str.len().gt(0) & frame["description"].str.len().ge(15) & known
    ].copy()
    frame["merchandising_group"] = frame["index_group_name"].replace(
        {"Ladieswear": "Ladieswear/Divided", "Divided": "Ladieswear/Divided"}
    )
    if frame["item_id"].duplicated().any():
        raise ValueError("H&M article IDs are not unique")
    texts: dict[str, str] = {}
    categories: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        texts[row.item_id] = (
            f"Title: {row.title[:200]} | Type: {row.product_type}"
            f" | Color: {row.colour_group_name} | Pattern: {row.graphical_appearance_name}"
            f" | Department: {row.index_name} | Section: {row.section_name}"
            f" | Description: {row.description[:500]}"
        )
        categories[row.item_id] = row.product_type
    return frame, texts, categories


def mine_hm_source(
    *,
    articles_path: str | Path,
    transactions_path: str | Path,
    manifest: dict,
    partitions: list[str],
    window_days: int,
    max_items: int,
    chunk_size: int = 250_000,
) -> tuple[list[Candidate], dict[str, str], dict[str, str]]:
    """Stream H&M purchases into compact integer arrays, then run the exact miner."""
    validate_manifest(manifest)
    catalog, texts, categories = canonical_hm_catalog(articles_path)
    requested = set(partitions)
    ordered_users = [
        str(user) for name in SPLIT_NAMES if name in requested for user in manifest["splits"][name]
    ]
    user_index = pd.Index(ordered_users)
    item_index = pd.Index(catalog["item_id"])
    partition_by_user = {str(user): name for name in requested for user in manifest["splits"][name]}
    user_chunks: list[np.ndarray] = []
    item_chunks: list[np.ndarray] = []
    day_chunks: list[np.ndarray] = []
    for frame in transaction_frames(transactions_path, chunk_size):
        required = {"customer_id", "article_id", "t_dat"}
        if missing := required - set(frame):
            raise ValueError(f"H&M transactions are missing columns: {sorted(missing)}")
        users = user_index.get_indexer(frame["customer_id"].astype(str))
        item_values = (
            pd.to_numeric(frame["article_id"], errors="raise")
            .astype("int64")
            .astype(str)
            .str.zfill(10)
        )
        items = item_index.get_indexer(item_values)
        dates = (
            pd.to_datetime(frame["t_dat"], errors="raise")
            .to_numpy()
            .astype("datetime64[D]")
            .astype(np.int32)
        )
        keep = (users >= 0) & (items >= 0)
        if keep.any():
            user_chunks.append(users[keep].astype(np.int32))
            item_chunks.append(items[keep].astype(np.int32))
            day_chunks.append(dates[keep])
    if not user_chunks:
        raise ValueError("No eligible H&M transactions in the selected partitions")
    candidates = mine_hm_encoded(
        np.concatenate(user_chunks),
        np.concatenate(item_chunks),
        np.concatenate(day_chunks),
        user_ids=user_index.to_numpy(),
        item_ids=catalog["item_id"].to_numpy(),
        product_codes=catalog["product_code"].to_numpy(),
        product_types=catalog["product_type"].to_numpy(),
        merchandising_groups=catalog["merchandising_group"].to_numpy(),
        user_partitions=np.array([partition_by_user[user] for user in user_index]),
        salt=str(manifest["salt"]),
        window_days=window_days,
        max_items=max_items,
    )
    return candidates, texts, categories

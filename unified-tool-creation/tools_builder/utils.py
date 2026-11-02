"""Shared utilities for the build pipeline."""

import json
import math
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import defaultdict
from datasets import get_dataset_config_info, load_dataset
from huggingface_hub import HfApi
from itertools import combinations

from tools_builder.storage import check_ephemeral_budget
from tools_builder.catalog_text import clean_description, first_available_year


AMAZON_REVISION = "2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e"


def save_config(args, data_dir: str, backbone: str):
    """Save current args to the config file, preserving any existing keys."""
    config_path = os.path.join(data_dir, f"config_{backbone}.json")
    existing = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            existing = json.load(f)
    existing.update(vars(args))
    from tools_builder.storage import atomic_json_dump
    atomic_json_dump(existing, config_path, indent=2)


def hits_at_k(ranks, ks):
    """Hit@K metrics. Ranks are 1-based (1 = best)."""
    return {k: (ranks <= k).float().mean().item() for k in ks}


def _append_qwen_eod(text: str) -> str:
    text = (text or "").rstrip()
    return text if text.endswith("<|endoftext|>") else f"{text}<|endoftext|>"


def encode_texts_base(
    texts, tokenizer, model,
    batch_size=64, max_length=256,
    prefix="", normalize=True,
    pooling="last_token", eod_token="<|endoftext|>",
):
    """Encode texts with a pretrained encoder. Supports last-token and mean pooling."""
    device = next(model.parameters()).device
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = [prefix + t for t in texts[i:i + batch_size]] if prefix else texts[i:i + batch_size]
        if eod_token:
            batch = [f"{t}{eod_token}" if not t.endswith(eod_token) else t for t in batch]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model(**inputs)
            if pooling == "last_token":
                if inputs['attention_mask'][:, -1].sum() == inputs['attention_mask'].shape[0]:
                    emb = output.last_hidden_state[:, -1]
                else:
                    seq_lens = inputs['attention_mask'].sum(dim=1) - 1
                    emb = output.last_hidden_state[
                        torch.arange(output.last_hidden_state.size(0), device=device), seq_lens
                    ]
            else:
                mask = inputs['attention_mask'].unsqueeze(-1).float()
                emb = (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

            if normalize:
                emb = F.normalize(emb, dim=-1)
            embeddings.append(emb.cpu().to(torch.float32))

    return torch.cat(embeddings)


def load_test_data(query_path, items, max_samples=500, seed=None):
    """Load query-item pairs for evaluation. Cache keys are 'user_id||item_id'."""
    with open(query_path) as f:
        cache = json.load(f)

    item_to_idx = {item: i for i, item in enumerate(items)}
    pairs = [
        (key.split("||")[1], query)
        for key, query in cache.items()
        if key.split("||")[1] in item_to_idx
    ]
    if seed is not None:
        import random
        random.seed(seed)
        random.shuffle(pairs)
    return [q for _, q in pairs[:max_samples]], [iid for iid, _ in pairs[:max_samples]], item_to_idx


def _detect_english(text):
    """Check if text is English using FastText."""
    from ftlangdetect import detect
    return detect(text.replace('\n', ' ')[:500])['lang'] == 'en'


def load_data(category):
    """Load Amazon Reviews 2023 dataset for a category."""
    if category == 'Electronics':
        from tools_builder.streamed_electronics import load_filtered_data
        return load_filtered_data()
    dataset_id = "McAuley-Lab/Amazon-Reviews-2023"
    configs = [f"raw_review_{category}", f"raw_meta_{category}"]
    expected_cache_bytes = 0
    try:
        for config in configs:
            info = get_dataset_config_info(
                dataset_id,
                config,
                revision=AMAZON_REVISION,
                trust_remote_code=True,
            )
            config_size = int(info.download_size or 0) + int(info.dataset_size or 0)
            if config_size <= 0:
                raise ValueError(f"Dataset config {config} did not report its size")
            expected_cache_bytes += config_size
    except Exception as error:
        # Some script-backed configs omit DatasetInfo sizes. Fall back to the
        # raw Hub objects and reserve 3x their compressed/source footprint for
        # source data plus Arrow conversion and temporary files.
        source_paths = [
            f"raw/review_categories/{category}.jsonl",
            f"raw/meta_categories/meta_{category}.jsonl",
        ]
        try:
            entries = HfApi().get_paths_info(
                dataset_id,
                source_paths,
                repo_type="dataset",
                revision=AMAZON_REVISION,
            )
            sizes = {entry.path: int(entry.size or 0) for entry in entries}
            missing = [path for path in source_paths if sizes.get(path, 0) <= 0]
            if missing:
                raise ValueError(f"Hub sizes unavailable for {missing}")
            expected_cache_bytes = 3 * sum(sizes.values())
            print(
                "Dataset config metadata omitted its footprint; using a "
                f"conservative Hub-source estimate of {expected_cache_bytes / 2**30:.2f} GiB."
            )
        except Exception as fallback_error:
            if os.getenv("RECO_ALLOW_UNKNOWN_DATASET_SIZE") != "1":
                raise RuntimeError(
                    "Could not determine the dataset cache footprint. Refusing an "
                    "unbounded download; set RECO_ALLOW_UNKNOWN_DATASET_SIZE=1 only "
                    "after manually verifying it fits the scratch budget."
                ) from fallback_error
            print(
                f"WARNING: dataset size preflight unavailable: {error}; "
                f"fallback failed: {fallback_error}"
            )
    else:
        pass

    if expected_cache_bytes:
        check_ephemeral_budget(
            expected_cache_bytes, reason=f"download dataset category {category}"
        )

    df_reviews = load_dataset(
        dataset_id,
        f"raw_review_{category}",
        split="full",
        revision=AMAZON_REVISION,
        trust_remote_code=True
    ).to_pandas()

    df_meta = load_dataset(
        dataset_id,
        f"raw_meta_{category}",
        split="full",
        revision=AMAZON_REVISION,
        trust_remote_code=True
    ).to_pandas()

    return df_reviews, df_meta


def merge_and_filter(
    df_rev, df_meta,
    min_reviews_per_user=1, min_reviews_per_item=1, max_users=0,
    price_range=(0.99, 1000.0), min_review_length=20, year_range=None,
):
    """Merge reviews with metadata and apply quality filters.

    Filter order optimized for efficiency (cheap filters first):
    1. Merge metadata (needed for year/price/description)
    2. Year range (cheap, drops many rows early)
    3. Price range (cheap)
    4. Description (clean + non-empty)
    5. Review length (on smaller set)
    6. English detection (expensive, on smallest set)
    7. User/item count filters
    8. Max users
    """
    df = df_rev.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop(columns=[c for c in ['asin', 'images'] if c in df.columns])

    column_renames = {
        'main_category': 'product_main_category',
        'title': 'product_title',
        'average_rating': 'product_average_rating',
        'rating_number': 'product_rating_number',
        'description': 'product_description',
        'price': 'product_price',
        'details': 'product_details'
    }
    df = df.merge(
        df_meta[['parent_asin'] + list(column_renames.keys())].rename(columns=column_renames),
        on='parent_asin', how='left'
    )

    df['product_year'] = df['product_details'].apply(first_available_year)
    df = df.drop(columns=['product_details'])
    if year_range:
        min_year, max_year = year_range
        df = df[df['product_year'].notna()]
        if min_year:
            df = df[df['product_year'] >= min_year]
        if max_year:
            df = df[df['product_year'] <= max_year]

    df['product_price'] = pd.to_numeric(df['product_price'].replace('None', pd.NA), errors='coerce')
    df = df[df['product_price'].notna() & (df['product_price'] >= price_range[0]) & (df['product_price'] <= price_range[1])]

    df['product_description'] = df['product_description'].apply(clean_description)
    df = df[df['product_description'].str.len() > 0]

    df['_t'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    df = df[df['_t'].str.len() >= min_review_length]

    df = df[df['_t'].apply(_detect_english)]
    df = df.drop(columns=['_t'])

    user_counts = df['user_id'].value_counts()
    df = df[df['user_id'].isin(user_counts[user_counts >= min_reviews_per_user].index)]

    if min_reviews_per_item > 1:
        item_counts = df['parent_asin'].value_counts()
        df = df[df['parent_asin'].isin(item_counts[item_counts >= min_reviews_per_item].index)]

    if max_users and df['user_id'].nunique() > max_users:
        top_users = df['user_id'].value_counts().head(max_users).index
        df = df[df['user_id'].isin(top_users)]

    return df.copy()


def compute_copurchase(df, min_rating=4.0, min_count=2, time_window_days=7):
    """Build co-purchase graph from positively-rated items by the same user."""
    df_positive = df[df['rating'] >= min_rating].drop_duplicates(subset=['user_id', 'parent_asin'])
    copurchase = defaultdict(lambda: defaultdict(int))

    if time_window_days is None:
        for items in df_positive.groupby('user_id')['parent_asin'].apply(set).values:
            for item_a, item_b in combinations(items, 2):
                copurchase[item_a][item_b] += 1
                copurchase[item_b][item_a] += 1
    else:
        user_data = df_positive.groupby('user_id').agg({'parent_asin': list, 'timestamp': list}).to_dict('index')
        for data in user_data.values():
            for (item_a, time_a), (item_b, time_b) in combinations(zip(data['parent_asin'], data['timestamp']), 2):
                if item_a != item_b and abs(time_a - time_b).days <= time_window_days:
                    copurchase[item_a][item_b] += 1
                    copurchase[item_b][item_a] += 1

    return {
        item: {other: count for other, count in neighbors.items() if count >= min_count}
        for item, neighbors in copurchase.items()
        if any(count >= min_count for count in neighbors.values())
    }


def filter_copurchase(copurchase, catalog_items):
    """Filter copurchase graph to catalog items only."""
    return {
        item: {other: count for other, count in neighbors.items() if other in catalog_items}
        for item, neighbors in copurchase.items()
        if item in catalog_items and any(other in catalog_items for other in neighbors)
    }


def compute_pmi(copurchase, min_pmi=0.0, min_count=1):
    """Convert raw co-purchase counts to Positive PMI scores.

    PMI(a,b) = log2(C(a,b) * N / (M(a) * M(b)))
    where N = total count mass, M(x) = marginal count for item x.
    Edges with raw count < *min_count* are dropped before PMI computation.
    Scores below *min_pmi* (default 0 = standard PPMI) are dropped.
    """
    filtered = {}
    for item, neighbors in copurchase.items():
        kept = {o: c for o, c in neighbors.items() if c >= min_count}
        if kept:
            filtered[item] = kept

    marginals = {}
    total = 0
    for item, neighbors in filtered.items():
        s = sum(neighbors.values())
        marginals[item] = s
        total += s

    if total == 0:
        return {}

    pmi_graph = {}
    for item, neighbors in filtered.items():
        m_a = marginals[item]
        scored = {}
        for other, count in neighbors.items():
            m_b = marginals.get(other, 0)
            if m_b == 0:
                continue
            pmi = math.log2(count * total / (m_a * m_b))
            if pmi >= min_pmi:
                scored[other] = round(pmi, 4)
        if scored:
            pmi_graph[item] = scored

    return pmi_graph


def compute_popularity(df_meta, catalog_items, product_reps):
    """Compute popularity rankings by subcategory (categories[-1])."""
    by_subcategory = defaultdict(list)

    for _, row in df_meta.iterrows():
        item_id = row['parent_asin']
        categories = row.get('categories', [])
        if item_id not in catalog_items or len(categories) == 0:
            continue

        by_subcategory[categories[-1]].append({
            'item_id': item_id,
            'review_count': int(row.get('rating_number', 0) or 0),
            'description': product_reps.get(item_id, '')[:300]
        })

    for subcategory in by_subcategory:
        by_subcategory[subcategory].sort(key=lambda x: -x['review_count'])

    return {'subcategories': sorted(by_subcategory.keys()), 'popularity': dict(by_subcategory)}


def compute_subcategory_mapping(df_meta, catalog_items):
    """Map items to their finest subcategory (categories[-1])."""
    item_to_subcategory = {}
    subcategory_to_items = defaultdict(list)

    for _, row in df_meta.iterrows():
        item_id = row['parent_asin']
        categories = row.get('categories', [])
        if item_id not in catalog_items or len(categories) == 0:
            continue

        subcategory = categories[-1]
        item_to_subcategory[item_id] = subcategory
        subcategory_to_items[subcategory].append(item_id)

    return {
        'item_to_subcategory': item_to_subcategory,
        'subcategory_to_items': dict(subcategory_to_items),
        'subcategories': sorted(subcategory_to_items.keys())
    }

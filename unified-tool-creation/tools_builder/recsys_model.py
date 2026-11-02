"""Multi-positive contrastive query-to-item retrieval training."""

import json
import math
import os
import random
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from tqdm import tqdm

from tools_builder.storage import atomic_json_dump, atomic_torch_save, check_ephemeral_budget
from tools_builder.retrieval import (
    OBJECTIVE_NAME,
    MultiPositiveQueryDataset,
    QueryGroup,
    build_query_groups,
    compute_multi_positive_retrieval_metrics,
    merge_positive_sets,
    mine_positive_aware_hard_negatives,
    multi_positive_collate,
    multi_positive_info_nce,
    normalize_query,
    query_group_fingerprint,
)

DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QWEN_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
DEFAULT_QUERY_INSTRUCTION = "Given a shopping query, retrieve relevant products."
QWEN_EOD_TOKEN = "<|endoftext|>"

BACKBONE_CONFIGS = {
    "qwen": {
        "model_name":   "Qwen/Qwen3-Embedding-0.6B",
        "revision":     QWEN_REVISION,
        "pooling":      "last_token",
        "query_prefix": "Instruct: {instruction}\nQuery: ",
        "item_prefix":  "",
        "eod_token":    "<|endoftext|>",
        "pad_side":     "left",
        "lora_targets": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    },
    "e5": {
        "model_name":   "intfloat/e5-base-v2",
        "revision":     None,
        "pooling":      "mean",
        "query_prefix": "query: ",
        "item_prefix":  "passage: ",
        "eod_token":    None,
        "pad_side":     "right",
        "lora_targets": [f"encoder.layer.{i}.output.dense" for i in range(12)],
    },
}


def format_query_for_instruction(query: str, instruction: str) -> str:
    return f"Instruct: {instruction}\nQuery: {query}"


def append_qwen_eod(text: str) -> str:
    text = (text or "").rstrip()
    return text if text.endswith(QWEN_EOD_TOKEN) else f"{text}{QWEN_EOD_TOKEN}"


def last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool final non-padding token for both left- and right-padded batches."""
    if attention_mask[:, -1].sum() == attention_mask.shape[0]:
        return last_hidden_state[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    return last_hidden_state[
        torch.arange(last_hidden_state.size(0), device=last_hidden_state.device), seq_lens
    ]


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class QueryRetrievalModel(nn.Module):
    """Dual-encoder model with LoRA fine-tuning on the query tower.

    The item tower is always frozen.  All embeddings are L2-normalized.
    """

    def __init__(
        self,
        backbone: str = "qwen",
        model_name: str = None,
        lora_r: int = 32,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        query_max_length: int = 256,
        item_max_length: int = 512,
        lora_target_modules: Optional[List[str]] = None,
        inference_only: bool = False,
        revision: str = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone_cfg = BACKBONE_CONFIGS[backbone]
        model_name = model_name or self.backbone_cfg["model_name"]
        self.model_name = model_name
        self.revision = revision or self.backbone_cfg.get("revision")
        self.query_instruction = query_instruction
        self.query_max_length = query_max_length
        self.item_max_length = item_max_length
        self.pooling = self.backbone_cfg["pooling"]
        pad_side = self.backbone_cfg["pad_side"]

        source_kwargs = {"revision": self.revision} if self.revision else {}
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side=pad_side, **source_kwargs
        )

        query_base = AutoModel.from_pretrained(model_name, **source_kwargs)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules or self.backbone_cfg["lora_targets"],
            bias="none",
        )
        self.query_encoder = get_peft_model(query_base, lora_config)

        if inference_only:
            self.base_encoder = None
        else:
            self.base_encoder = AutoModel.from_pretrained(model_name, **source_kwargs).eval()
            for param in self.base_encoder.parameters():
                param.requires_grad = False

        models = [self.query_encoder] + ([self.base_encoder] if self.base_encoder else [])
        self._ensure_padding_token(self.tokenizer, *models)

    @staticmethod
    def _ensure_padding_token(tokenizer, *models):
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is not None:
            for model in models:
                model.config.pad_token_id = tokenizer.pad_token_id

    def train(self, mode=True):
        super().train(mode)
        if mode and self.base_encoder is not None:
            self.base_encoder.eval()
        return self

    def _pool(self, output, attention_mask):
        if self.pooling == "last_token":
            pooled = last_token_pool(output.last_hidden_state, attention_mask)
        else:
            pooled = mean_pool(output.last_hidden_state, attention_mask)
        return F.normalize(pooled, p=2, dim=-1)

    def _format_query(self, text: str) -> str:
        prefix = self.backbone_cfg["query_prefix"]
        if "{instruction}" in prefix:
            prefix = prefix.format(instruction=self.query_instruction)
        eod = self.backbone_cfg["eod_token"] or ""
        return f"{prefix}{text}{eod}"

    def _format_item(self, text: str) -> str:
        prefix = self.backbone_cfg["item_prefix"]
        eod = self.backbone_cfg["eod_token"] or ""
        return f"{prefix}{(text or '').rstrip()}{eod}"

    def _encode(self, texts: List[str], encoder, format_fn, max_length: int) -> torch.Tensor:
        device = next(encoder.parameters()).device
        inputs = self.tokenizer(
            [format_fn(t) for t in texts],
            padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(device)
        return self._pool(encoder(**inputs), inputs['attention_mask'])

    def encode_queries(self, texts: List[str]) -> torch.Tensor:
        return self._encode(texts, self.query_encoder, self._format_query, self.query_max_length)

    def encode_items(self, texts: List[str]) -> torch.Tensor:
        return self._encode(texts, self.base_encoder, self._format_item, self.item_max_length)

    def encode_base_queries(self, texts: List[str]) -> torch.Tensor:
        """Encode queries with frozen base model for baseline comparison."""
        return self._encode(texts, self.base_encoder, self._format_query, self.query_max_length)


def split_queries_by_product(
    queries: Dict[Tuple[str, str], str],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Dict, Dict]:
    """Create deterministic train/validation splits with no product overlap."""
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be strictly between 0 and 1")

    products = sorted({item_id for _, item_id in queries})
    if len(products) < 2:
        raise ValueError("Product-disjoint splitting requires at least two products")

    random.Random(seed).shuffle(products)
    val_product_count = min(
        len(products) - 1,
        max(1, int(len(products) * val_ratio)),
    )
    val_products = set(products[:val_product_count])

    train_queries, val_queries = {}, {}
    for key, query in queries.items():
        target = val_queries if key[1] in val_products else train_queries
        target[key] = query

    return train_queries, val_queries


def compute_subcat_match(query_embs: torch.Tensor, item_embs: torch.Tensor,
                         item_ids: List[str], target_subcats: List[str],
                         item_to_subcat: Dict[str, str], ks: List[int] = [5, 10]) -> Dict[int, float]:
    """Compute % of top-K retrieved items matching the query's target subcategory.

    Processes in chunks to avoid OOM on large catalogs.
    """
    max_k = max(ks)
    totals = {k: 0.0 for k in ks}
    valid = 0
    chunk_size = 256
    idx = 0

    for i in range(0, len(query_embs), chunk_size):
        chunk_q = query_embs[i:i + chunk_size]
        scores = chunk_q @ item_embs.T
        topk_idx = scores.topk(min(max_k, scores.shape[1]), dim=1).indices

        for j in range(len(chunk_q)):
            target_sc = target_subcats[idx]
            idx += 1
            if not target_sc:
                continue
            valid += 1
            for k in ks:
                top_items = [item_ids[ti] for ti in topk_idx[j, :k].tolist()]
                totals[k] += sum(1 for it in top_items if item_to_subcat.get(it) == target_sc) / k

    return {k: v / max(valid, 1) for k, v in totals.items()}


def compute_retrieval_metrics(
    query_embs: torch.Tensor,
    item_embs: torch.Tensor,
    item_ids: List[str],
    target_item_ids: List[str],
    ks: Tuple[int, ...] = (1, 5, 10),
    chunk_size: int = 256,
    target_subcategories: Optional[List[str]] = None,
    item_to_subcategory: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """Rank the exact held-out target product against the full catalog."""
    if query_embs.ndim != 2 or item_embs.ndim != 2:
        raise ValueError("Query and item embeddings must be rank-2 tensors")
    if len(query_embs) != len(target_item_ids):
        raise ValueError("Each query embedding must have one target item ID")
    if len(item_embs) != len(item_ids):
        raise ValueError("Each item embedding must have one item ID")
    if query_embs.shape[1] != item_embs.shape[1]:
        raise ValueError("Query and item embedding dimensions do not match")
    if target_subcategories is not None and len(target_subcategories) != len(query_embs):
        raise ValueError("Each query must have one target subcategory")
    if (target_subcategories is None) != (item_to_subcategory is None):
        raise ValueError(
            "target_subcategories and item_to_subcategory must be supplied together"
        )

    item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
    valid_rows = [idx for idx, item_id in enumerate(target_item_ids) if item_id in item_to_idx]
    if not valid_rows:
        raise ValueError("No validation target products exist in the evaluation catalog")

    ranks = []
    subcategory_totals = {k: 0.0 for k in ks}
    valid_subcategories = 0
    max_k = min(max(ks), len(item_ids))
    for start in range(0, len(valid_rows), chunk_size):
        rows = valid_rows[start:start + chunk_size]
        chunk_q = query_embs[rows].to(
            device=item_embs.device,
            dtype=item_embs.dtype,
        )
        scores = chunk_q @ item_embs.T
        target_idx = torch.tensor(
            [item_to_idx[target_item_ids[row]] for row in rows],
            dtype=torch.long,
            device=scores.device,
        )
        target_scores = scores.gather(1, target_idx.unsqueeze(1))
        # Strictly greater scores give a deterministic, optimistic tie policy;
        # exact floating-point ties are exceptionally rare for dense embeddings.
        ranks.append(1 + (scores > target_scores).sum(dim=1).cpu())
        if target_subcategories is not None:
            topk_idx = scores.topk(max_k, dim=1).indices.cpu().tolist()
            for local_row, source_row in enumerate(rows):
                target_subcategory = target_subcategories[source_row]
                if not target_subcategory:
                    continue
                valid_subcategories += 1
                for k in ks:
                    effective_k = min(k, len(item_ids))
                    matching = sum(
                        item_to_subcategory.get(item_ids[item_idx]) == target_subcategory
                        for item_idx in topk_idx[local_row][:effective_k]
                    )
                    subcategory_totals[k] += matching / effective_k

    ranks_tensor = torch.cat(ranks).to(torch.float32)
    metrics = {
        f"hit_at_{k}": (ranks_tensor <= k).to(torch.float32).mean().item()
        for k in ks
    }
    metrics.update({
        "mrr": ranks_tensor.reciprocal().mean().item(),
        "mean_rank": ranks_tensor.mean().item(),
        "median_rank": ranks_tensor.median().item(),
        "valid_queries": int(len(ranks_tensor)),
        "catalog_items": int(len(item_ids)),
    })
    if target_subcategories is not None:
        for k in ks:
            metrics[f"subcategory_match_at_{k}"] = (
                subcategory_totals[k] / max(valid_subcategories, 1)
            )
    return metrics


def batch_encode(encode_fn, texts, batch_size, desc):
    """Encode texts in chunks, returning concatenated CPU embeddings."""
    embs = []
    with torch.no_grad(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()
    ):
        for i in tqdm(range(0, len(texts), batch_size), desc=desc):
            embs.append(encode_fn(texts[i:i + batch_size]).cpu())
    if not embs:
        raise ValueError(f"Cannot encode an empty collection: {desc}")
    return torch.cat(embs)


def _expand_query_embeddings(
    unique_embeddings: torch.Tensor,
    unique_queries: List[str],
    expanded_queries: List[str],
) -> torch.Tensor:
    lookup = {query: index for index, query in enumerate(unique_queries)}
    indices = torch.tensor([lookup[normalize_query(query)] for query in expanded_queries])
    return unique_embeddings.index_select(0, indices)


def _candidate_metrics_from_embeddings(
    query_embeddings,
    loader,
    item_embeddings,
    temperature,
):
    """Score deterministic candidate sets without encoding validation twice."""
    totals = {}
    examples = 0
    offset = 0
    with torch.no_grad():
        for batch in loader:
            batch_size = len(batch["queries"])
            batch_embeddings = query_embeddings[offset:offset + batch_size].to(
                device=item_embeddings.device,
                dtype=item_embeddings.dtype,
            )
            offset += batch_size
            loss, metrics = multi_positive_info_nce(
                batch_embeddings,
                item_embeddings,
                batch["positive_indices"],
                batch["negative_indices"],
                temperature=temperature,
            )
            metrics = dict(metrics)
            metrics["contrastive_loss"] = loss.item()
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    totals[key] = totals.get(key, 0.0) + float(value) * batch_size
            examples += batch_size
    if offset != len(query_embeddings):
        raise ValueError("Validation embeddings and candidate rows are misaligned")
    return {key: value / max(examples, 1) for key, value in totals.items()}


def train_query_retrieval(
    train_queries: Dict[Tuple[str, str], str],
    val_queries: Dict[Tuple[str, str], str],
    product_reps: Dict[str, str],
    item_to_subcategory: Dict[str, str],
    *,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 7.406576281631978e-5,
    lora_r: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    patience: int = 2,
    save_dir: str = "outputs",
    seed: int = 42,
    backbone: str = "qwen",
    model_name: str = None,
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
    query_max_length: int = 256,
    item_max_length: int = 512,
    lora_target_modules: Optional[List[str]] = None,
    grad_accum_steps: int = 2,
    lr_schedule: str = "constant",
    warmup_ratio: float = 0.0,
    temperature: float = 0.020090056606207288,
    hard_negatives_per_query: int = 4,
    random_negatives_per_query: int = 4,
    hard_negative_candidates: int = 100,
    hard_negative_positive_ratio: float = 0.95,
    early_stop_metric: str = "val_multi_positive_mrr",
    checkpoint_callback=None,
    baseline_callback=None,
    mining_callback=None,
    resume_from_checkpoint: bool = False,
    catalog_item_embeddings: Optional[torch.Tensor] = None,
    hard_negatives_path: Optional[str] = None,
    require_product_disjoint: bool = True,
):
    """Train query LoRA with multi-positive InfoNCE and cached item vectors."""
    if catalog_item_embeddings is None:
        raise ValueError("Retrieval training requires cached catalog item embeddings")
    os.makedirs(save_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    all_item_ids = list(product_reps)
    item_to_index = {item_id: index for index, item_id in enumerate(all_item_ids)}
    train_product_ids = {item_id for _, item_id in train_queries}
    val_product_ids = {item_id for _, item_id in val_queries}
    overlap = train_product_ids & val_product_ids
    if overlap and require_product_disjoint:
        raise ValueError(f"Train/validation product leakage detected: {len(overlap)} items")

    train_groups = build_query_groups(train_queries, train_product_ids)
    val_groups = build_query_groups(val_queries, val_product_ids)
    all_known_positives = merge_positive_sets(train_groups, val_groups)
    val_eval_groups = [
        QueryGroup(
            group.query,
            tuple(sorted(all_known_positives[group.query])),
            group.occurrences,
        )
        for group in val_groups
    ]
    train_query_fingerprint = query_group_fingerprint(train_groups)
    print(
        "Balanced multi-positive queries: "
        f"{len(train_queries):,} rows -> {len(train_groups):,} train anchors; "
        f"{len(val_queries):,} rows -> {len(val_groups):,} validation anchors."
    )
    print(
        f"Known query/product collisions: "
        f"{sum(len(items) > 1 for items in all_known_positives.values()):,} queries."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved_model = model_name or BACKBONE_CONFIGS[backbone]["model_name"]
    print(f"Loading QueryRetrievalModel ({resolved_model}) on {device} for {OBJECTIVE_NAME}...")
    model = QueryRetrievalModel(
        backbone=backbone,
        model_name=model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        query_instruction=query_instruction,
        query_max_length=query_max_length,
        item_max_length=item_max_length,
        lora_target_modules=lora_target_modules,
    ).to(device)
    model.query_encoder.gradient_checkpointing_enable()
    model.query_encoder.enable_input_require_grads()
    item_embeddings = F.normalize(
        catalog_item_embeddings.detach().to(device=device, dtype=torch.float32), dim=-1
    )
    if len(item_embeddings) != len(all_item_ids):
        raise ValueError("Cached item embeddings do not match the product catalog")

    # Encode each distinct validation query once. Exact-target metrics are
    # expanded afterward, while multi-positive metrics stay query-balanced.
    val_unique_queries = [group.query for group in val_groups]
    # Held-out relevance remains product-disjoint for checkpoint selection.
    # A secondary all-known metric reflects shopper utility when a train-split
    # product is also a valid result for the same broad query.
    val_relevant_items = [group.positive_item_ids for group in val_groups]
    val_all_known_relevant_items = [
        tuple(sorted(all_known_positives[group.query])) for group in val_groups
    ]
    val_rows = list(val_queries.items())
    val_row_queries = [normalize_query(query) for _, query in val_rows]
    val_target_item_ids = [key[1] for key, _ in val_rows]
    val_target_subcats = [item_to_subcategory.get(item_id) for item_id in val_target_item_ids]

    model.query_encoder.to("cpu")
    torch.cuda.empty_cache()
    print(f"Encoding {len(val_unique_queries):,} unique validation queries for baseline...")
    base_unique_query_embeddings = batch_encode(
        model.encode_base_queries, val_unique_queries, 64, "Base unique query embeddings"
    )
    base_exact_query_embeddings = _expand_query_embeddings(
        base_unique_query_embeddings, val_unique_queries, val_row_queries
    )
    base_retrieval = compute_retrieval_metrics(
        base_exact_query_embeddings.to(device),
        item_embeddings,
        all_item_ids,
        val_target_item_ids,
        target_subcategories=val_target_subcats,
        item_to_subcategory=item_to_subcategory,
    )
    base_multi_retrieval = compute_multi_positive_retrieval_metrics(
        base_unique_query_embeddings.to(device),
        item_embeddings,
        all_item_ids,
        val_relevant_items,
    )
    base_all_known_retrieval = compute_multi_positive_retrieval_metrics(
        base_unique_query_embeddings.to(device),
        item_embeddings,
        all_item_ids,
        val_all_known_relevant_items,
    )
    del base_exact_query_embeddings
    torch.cuda.empty_cache()

    # Mine globally confusing products once from the frozen base model. The
    # artifact contains IDs only and is cheap to persist/resume from HF.
    hard_negatives = None
    if hard_negatives_path and os.path.exists(hard_negatives_path):
        with open(hard_negatives_path) as handle:
            payload = json.load(handle)
        config = payload.get("config", {})
        if (
            payload.get("objective") == OBJECTIVE_NAME
            and config.get("query_fingerprint") == train_query_fingerprint
            and config.get("top_k") == hard_negative_candidates
            and config.get("negatives_per_query") == hard_negatives_per_query
            and config.get("positive_ratio") == hard_negative_positive_ratio
        ):
            hard_negatives = payload.get("negatives", {})
            print(f"Restored {len(hard_negatives):,} mined query-negative sets.")
    if hard_negatives is None:
        print("Mining positive-aware global hard negatives...")
        hard_negatives = mine_positive_aware_hard_negatives(
            model.encode_base_queries,
            train_groups,
            item_embeddings,
            all_item_ids,
            train_product_ids,
            top_k=hard_negative_candidates,
            negatives_per_query=hard_negatives_per_query,
            positive_ratio=hard_negative_positive_ratio,
            batch_size=64,
        )
        if hard_negatives_path:
            payload = {
                "objective": OBJECTIVE_NAME,
                "config": {
                    "query_count": len(train_groups),
                    "query_fingerprint": train_query_fingerprint,
                    "top_k": hard_negative_candidates,
                    "negatives_per_query": hard_negatives_per_query,
                    "positive_ratio": hard_negative_positive_ratio,
                },
                "negatives": hard_negatives,
            }
            atomic_json_dump(payload, hard_negatives_path)
            if mining_callback:
                mining_callback(hard_negatives_path)

    train_dataset = MultiPositiveQueryDataset(
        train_groups,
        item_to_index,
        train_product_ids,
        hard_negatives=hard_negatives,
        hard_negatives_per_query=hard_negatives_per_query,
        random_negatives_per_query=random_negatives_per_query,
        seed=seed,
        dynamic=True,
    )
    val_dataset = MultiPositiveQueryDataset(
        val_eval_groups,
        item_to_index,
        all_item_ids,
        hard_negatives_per_query=0,
        random_negatives_per_query=max(8, random_negatives_per_query),
        seed=seed,
        dynamic=False,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=multi_positive_collate,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=multi_positive_collate,
    )
    baseline_candidate = _candidate_metrics_from_embeddings(
        base_unique_query_embeddings, val_loader, item_embeddings, temperature
    )

    metrics_dir = os.path.join(save_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    baseline_path = os.path.join(metrics_dir, f"retrieval_baseline_{backbone}.json")
    baseline_report = {
        "model": resolved_model,
        "backbone": backbone,
        "objective": OBJECTIVE_NAME,
        "split": (
            "product_disjoint_validation"
            if require_product_disjoint
            else "externally_edge_disjoint_validation"
        ),
        "train_products": len(train_product_ids),
        "validation_products": len(val_product_ids),
        "validation_rows": len(val_queries),
        "validation_unique_queries": len(val_groups),
        "retrieval": base_retrieval,
        "multi_positive_retrieval": base_multi_retrieval,
        "all_known_relevance_retrieval": base_all_known_retrieval,
        "candidate_metrics": baseline_candidate,
        # Backward-compatible field name; this now uses collision-aware candidates.
        "pairwise_accuracy": baseline_candidate["pairwise_accuracy"],
    }
    atomic_json_dump(baseline_report, baseline_path, indent=2)
    if baseline_callback:
        baseline_callback(baseline_path)
    print(
        "Untrained baseline: "
        f"exact MRR={base_retrieval['mrr']:.4f}, "
        f"multi-positive MRR={base_multi_retrieval['mrr']:.4f}, "
        f"multi nDCG@10={base_multi_retrieval['ndcg_at_10']:.2%}."
    )

    model.base_encoder.to("cpu")
    model.query_encoder.to(device)
    torch.cuda.empty_cache()
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    total_training_steps = optimizer_steps_per_epoch * epochs
    warmup_steps = int(total_training_steps * warmup_ratio)
    if lr_schedule == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )
    elif lr_schedule == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
        )
    else:
        raise ValueError(f"Unsupported retrieval LR schedule: {lr_schedule}")
    resolved_targets = lora_target_modules or BACKBONE_CONFIGS[backbone]["lora_targets"]
    model_config = {
        "objective": OBJECTIVE_NAME,
        "backbone": backbone,
        "model_name": model.model_name,
        "revision": model.revision,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_target_modules": resolved_targets,
        "query_instruction": query_instruction,
        "query_max_length": query_max_length,
        "item_max_length": item_max_length,
        "temperature": temperature,
        "learning_rate": lr,
        "lr_schedule": lr_schedule,
        "warmup_ratio": warmup_ratio,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "epochs": epochs,
        "require_product_disjoint": require_product_disjoint,
    }

    models_dir = os.path.join(save_dir, "models")
    checkpoint_dir = os.path.join(models_dir, f"recsys_checkpoints_{backbone}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    best_path = os.path.join(models_dir, f"recsys_model_{backbone}.pt")
    history_path = os.path.join(metrics_dir, f"retrieval_history_{backbone}.json")

    baseline_metric_by_name = {
        "val_multi_positive_mrr": base_multi_retrieval["mrr"],
        "val_retrieval_mrr": base_retrieval["mrr"],
        "val_multi_positive_ndcg_at_10": base_multi_retrieval["ndcg_at_10"],
    }
    if early_stop_metric not in baseline_metric_by_name:
        raise ValueError(
            f"Unsupported retrieval early-stop metric {early_stop_metric!r}; choose from "
            + ", ".join(sorted(baseline_metric_by_name))
        )
    best_metric = baseline_metric_by_name[early_stop_metric]
    patience_counter = 0
    history = []
    start_epoch = 0

    initial_adapter = {
        key: value.detach().cpu()
        for key, value in get_peft_model_state_dict(model.query_encoder).items()
    }
    adapter_bytes = sum(value.numel() * value.element_size() for value in initial_adapter.values())
    expected_checkpoint_bytes = max(16 * 1024 * 1024, adapter_bytes * 4)
    baseline_checkpoint = {
        "epoch": 0,
        "adapter_state_dict": initial_adapter,
        "model_config": model_config,
        early_stop_metric: best_metric,
        "best_metric": best_metric,
        "patience_counter": 0,
        "history": [],
        "baseline_winner": True,
    }
    has_resume_checkpoint = resume_from_checkpoint and os.path.exists(latest_path)
    if has_resume_checkpoint and not os.path.exists(best_path):
        raise FileNotFoundError(
            "A resumable latest checkpoint exists but its best checkpoint is missing"
        )
    if not has_resume_checkpoint:
        atomic_torch_save(
            baseline_checkpoint, best_path, expected_bytes=expected_checkpoint_bytes
        )
        if checkpoint_callback:
            # The untrained encoder is a real candidate in baseline-aware model
            # selection. Persist it before epoch 1 so an interrupted run always
            # has both the latest state and the winning state needed to resume.
            checkpoint_callback([best_path], 0, True)

    if has_resume_checkpoint:
        resume = torch.load(latest_path, map_location="cpu", weights_only=False)
        if resume.get("model_config", {}).get("objective") != OBJECTIVE_NAME:
            raise ValueError(
                "The restored checkpoint uses the unsupported historical BPR objective. "
                "Use a new run_id for multi-positive retrieval training."
            )
        resume_config = resume.get("model_config", {})
        incompatible = {
            key: (resume_config[key], value)
            for key, value in model_config.items()
            if key in resume_config and resume_config[key] != value
        }
        if incompatible:
            details = ", ".join(
                f"{key}: checkpoint={old!r}, requested={new!r}"
                for key, (old, new) in sorted(incompatible.items())
            )
            raise ValueError(
                "Refusing to resume a retrieval checkpoint with different "
                f"hyperparameters ({details}). Use a new run_id."
            )
        set_peft_model_state_dict(model.query_encoder, resume["adapter_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        history = resume.get("history", [])
        start_epoch = int(resume.get("epoch", 0))
        best_metric = float(resume.get("best_metric", best_metric))
        patience_counter = int(resume.get("patience_counter", 0))
        print(f"Resuming retrieval at epoch {start_epoch + 1}.")

    for epoch in range(start_epoch, epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()
        train_totals = {"loss": 0.0, "pairwise": 0.0, "top1": 0.0}
        progress = tqdm(train_loader, desc=f"Retrieval epoch {epoch + 1}/{epochs}")
        for step, batch in enumerate(progress):
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                query_embeddings = model.encode_queries(batch["queries"])
                loss, metrics = multi_positive_info_nce(
                    query_embeddings,
                    item_embeddings,
                    batch["positive_indices"],
                    batch["negative_indices"],
                    temperature=temperature,
                )
            (loss / grad_accum_steps).backward()
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            train_totals["loss"] += loss.item()
            train_totals["pairwise"] += metrics["pairwise_accuracy"]
            train_totals["top1"] += metrics["candidate_top1_accuracy"]
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                candidate_top1=f"{metrics['candidate_top1_accuracy']:.2%}",
            )
        train_avg = {key: value / len(train_loader) for key, value in train_totals.items()}

        model.eval()
        fine_unique_embeddings = batch_encode(
            model.encode_queries,
            val_unique_queries,
            64,
            f"Retrieval epoch {epoch + 1} unique query embeddings",
        )
        val_candidate = _candidate_metrics_from_embeddings(
            fine_unique_embeddings, val_loader, item_embeddings, temperature
        )
        fine_exact_embeddings = _expand_query_embeddings(
            fine_unique_embeddings, val_unique_queries, val_row_queries
        )
        retrieval = compute_retrieval_metrics(
            fine_exact_embeddings.to(device),
            item_embeddings,
            all_item_ids,
            val_target_item_ids,
            target_subcategories=val_target_subcats,
            item_to_subcategory=item_to_subcategory,
        )
        multi_retrieval = compute_multi_positive_retrieval_metrics(
            fine_unique_embeddings.to(device),
            item_embeddings,
            all_item_ids,
            val_relevant_items,
        )
        all_known_retrieval = compute_multi_positive_retrieval_metrics(
            fine_unique_embeddings.to(device),
            item_embeddings,
            all_item_ids,
            val_all_known_relevant_items,
        )
        del fine_exact_embeddings
        torch.cuda.empty_cache()

        epoch_metrics = {
            "epoch": epoch + 1,
            "objective": OBJECTIVE_NAME,
            "train_contrastive_loss": train_avg["loss"],
            "train_candidate_pairwise_accuracy": train_avg["pairwise"],
            "train_candidate_top1_accuracy": train_avg["top1"],
            "val_contrastive_loss": val_candidate["contrastive_loss"],
            "val_candidate_pairwise_accuracy": val_candidate["pairwise_accuracy"],
            "val_candidate_top1_accuracy": val_candidate["candidate_top1_accuracy"],
            "val_retrieval_hit_at_1": retrieval["hit_at_1"],
            "val_retrieval_hit_at_5": retrieval["hit_at_5"],
            "val_retrieval_hit_at_10": retrieval["hit_at_10"],
            "val_retrieval_mrr": retrieval["mrr"],
            "val_retrieval_mean_rank": retrieval["mean_rank"],
            "val_retrieval_median_rank": retrieval["median_rank"],
            "val_subcategory_match_at_5": retrieval["subcategory_match_at_5"],
            "val_subcategory_match_at_10": retrieval["subcategory_match_at_10"],
            "val_multi_positive_hit_at_1": multi_retrieval["hit_at_1"],
            "val_multi_positive_hit_at_5": multi_retrieval["hit_at_5"],
            "val_multi_positive_hit_at_10": multi_retrieval["hit_at_10"],
            "val_multi_positive_recall_at_10": multi_retrieval["recall_at_10"],
            "val_multi_positive_ndcg_at_10": multi_retrieval["ndcg_at_10"],
            "val_multi_positive_mrr": multi_retrieval["mrr"],
            "val_all_known_relevance_mrr": all_known_retrieval["mrr"],
            "val_all_known_relevance_ndcg_at_10": all_known_retrieval["ndcg_at_10"],
            "baseline_retrieval_mrr": base_retrieval["mrr"],
            "baseline_retrieval_hit_at_1": base_retrieval["hit_at_1"],
            "baseline_retrieval_hit_at_5": base_retrieval["hit_at_5"],
            "baseline_retrieval_hit_at_10": base_retrieval["hit_at_10"],
            "baseline_multi_positive_mrr": base_multi_retrieval["mrr"],
            "baseline_multi_positive_ndcg_at_10": base_multi_retrieval["ndcg_at_10"],
            "learning_rate": float(f"{optimizer.param_groups[0]['lr']:.6e}"),
        }
        history.append(epoch_metrics)
        atomic_json_dump(
            {"baseline": baseline_report, "epochs": history}, history_path, indent=2
        )
        print(
            f"Retrieval epoch {epoch + 1}: exact Hit@1/5/10="
            f"{retrieval['hit_at_1']:.2%}/{retrieval['hit_at_5']:.2%}/{retrieval['hit_at_10']:.2%}, "
            f"MRR={retrieval['mrr']:.4f}; multi-positive MRR={multi_retrieval['mrr']:.4f}, "
            f"nDCG@10={multi_retrieval['ndcg_at_10']:.2%} "
            f"(baseline {base_multi_retrieval['mrr']:.4f}/{base_multi_retrieval['ndcg_at_10']:.2%})."
        )

        current_metric = epoch_metrics[early_stop_metric]
        is_best = current_metric > best_metric
        if is_best:
            best_metric = current_metric
            patience_counter = 0
        else:
            patience_counter += 1
        adapter_state = {
            key: value.detach().cpu()
            for key, value in get_peft_model_state_dict(model.query_encoder).items()
        }
        checkpoint = {
            "epoch": epoch + 1,
            "adapter_state_dict": adapter_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config": model_config,
            early_stop_metric: current_metric,
            "best_metric": best_metric,
            "patience_counter": patience_counter,
            "history": history,
            "baseline_winner": not is_best and best_metric == baseline_metric_by_name[early_stop_metric],
        }
        check_ephemeral_budget(expected_checkpoint_bytes * 2, reason="rotating retrieval LoRA checkpoint")
        atomic_torch_save(checkpoint, latest_path, expected_bytes=expected_checkpoint_bytes)
        uploaded_paths = [latest_path, history_path]
        if is_best:
            atomic_torch_save(checkpoint, best_path, expected_bytes=expected_checkpoint_bytes)
            uploaded_paths.append(best_path)
        if checkpoint_callback:
            checkpoint_callback(uploaded_paths, epoch + 1, is_best)
        if patience_counter >= patience:
            print(
                f"Baseline-aware early stopping at epoch {epoch + 1}: "
                f"{early_stop_metric} has not exceeded {best_metric:.4f}."
            )
            break

    atomic_json_dump({"baseline": baseline_report, "epochs": history}, history_path, indent=2)
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    set_peft_model_state_dict(model.query_encoder, best_checkpoint["adapter_state_dict"])
    winner = "untrained baseline" if best_checkpoint.get("epoch") == 0 else f"epoch {best_checkpoint['epoch']}"
    print(f"Restored retrieval winner: {winner} ({early_stop_metric}={best_metric:.4f}).")
    return model, history

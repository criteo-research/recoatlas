"""H&M embedding-space residual complementarity trainer used in the paper."""

from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from tools_builder.stage4_complementarity import (
    BalancedComplementarityDataset,
    build_temporal_complementarity_split,
    complementarity_collate,
    compute_complementarity_ranking_metrics,
    multi_positive_complementarity_loss,
)
from tools_builder.storage import atomic_json_dump, atomic_torch_save


class ResidualHead(nn.Module):
    def __init__(self, dim: int, rank: int = 32, alpha: float = 32.0, dropout: float = 0.1):
        super().__init__()
        self.a = nn.Linear(dim, rank, bias=False)
        self.b = nn.Linear(rank, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank
        nn.init.zeros_(self.b.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = self.b(self.a(self.dropout(values)))
        return F.normalize(values + self.scale * update, dim=-1)


class ResidualComplementarity(nn.Module):
    """Independent anchor/complement heads over frozen catalog embeddings."""

    def __init__(
        self,
        dim: int,
        rank: int = 32,
        alpha: float = 32.0,
        dropout: float = 0.1,
        temperature: float = 0.081621255477171,
    ):
        super().__init__()
        self.anchor_head = ResidualHead(dim, rank, alpha, dropout)
        self.complement_head = ResidualHead(dim, rank, alpha, dropout)
        self.temperature = temperature

    def project_anchor(self, values: torch.Tensor) -> torch.Tensor:
        return self.anchor_head(values)

    def project_complement(self, values: torch.Tensor) -> torch.Tensor:
        return self.complement_head(values)


def _project_catalog(model, base, *, chunk_size=4096):
    model.eval()
    with torch.no_grad():
        anchors = torch.cat([model.project_anchor(chunk).cpu() for chunk in base.split(chunk_size)])
        complements = torch.cat(
            [model.project_complement(chunk).cpu() for chunk in base.split(chunk_size)]
        )
    return anchors.float(), complements.float()


def train_residual_complementarity(
    save_dir: str | Path,
    *,
    epochs: int = 20,
    selected_epoch: int = 2,
    rank: int = 32,
    alpha: float = 32.0,
    dropout: float = 0.1,
    temperature: float = 0.081621255477171,
    learning_rate: float = 0.00040792119058488896,
    batch_size: int = 1024,
    positives_per_anchor: int = 8,
    random_negatives_per_anchor: int = 8,
    seed: int = 42,
) -> dict:
    """Train all epochs and export the paper's explicitly selected epoch 2."""
    if not 1 <= selected_epoch <= epochs:
        raise ValueError("selected_epoch must be inside the training range")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for H&M residual complementarity training")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    save_dir = Path(save_dir)
    data_dir = save_dir / "data"
    embedding_dir = save_dir / "embeddings"
    model_dir = save_dir / "models" / "comp_residual_checkpoints_qwen"
    metrics_dir = save_dir / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    product_texts = json.loads((data_dir / "product_texts.json").read_text())
    subcategories = json.loads((data_dir / "subcategories.json").read_text())[
        "item_to_subcategory"
    ]
    interactions = pd.read_parquet(data_dir / "interactions.parquet")
    embedding_payload = torch.load(
        embedding_dir / "item_embeddings_qwen.pt", map_location="cpu", weights_only=True
    )
    item_ids = embedding_payload["item_ids"]
    if item_ids != list(product_texts):
        raise ValueError("H&M item embeddings are not aligned with product_texts.json")

    split_path = data_dir / "comp_split.pt"
    if split_path.exists():
        split = torch.load(split_path, map_location="cpu", weights_only=False)
    else:
        split = build_temporal_complementarity_split(
            interactions,
            item_ids,
            subcategories,
            window_days=7,
            min_rating=4.0,
            min_train_count=1,
            min_pmi=0.0,
            cross_subcategory=True,
            validation_user_ratio=0.1,
            seed=seed,
        )
        atomic_torch_save(split, split_path)
    if split.metadata.get("post_split_edge_overlap") != 0:
        raise ValueError("Complementarity split contains held-out training edges")

    device = torch.device("cuda")
    base = embedding_payload["item_embs"].float().to(device)
    model = ResidualComplementarity(
        base.shape[1], rank, alpha, dropout, temperature
    ).to(device)
    expected = F.normalize(base, dim=-1)
    model.train()
    with torch.no_grad():
        if not torch.equal(model.project_anchor(base[:1024]), expected[:1024]):
            raise AssertionError("Residual initialization must equal the base embedding")

    dataset = BalancedComplementarityDataset(
        split.train_graph,
        {item: index for index, item in enumerate(item_ids)},
        subcategories,
        positives_per_anchor=positives_per_anchor,
        hard_negatives_per_anchor=0,
        random_negatives_per_anchor=random_negatives_per_anchor,
        cross_subcategory=True,
        seed=seed,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=complementarity_collate
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = []
    selected_checkpoint = None

    for epoch_index in range(epochs):
        epoch = epoch_index + 1
        dataset.set_epoch(epoch_index)
        dataset.set_hard_negatives({})
        model.train()
        losses = []
        accuracies = []
        for batch in loader:
            optimizer.zero_grad()
            loss, row = multi_positive_complementarity_loss(model, base, batch)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite residual complementarity loss")
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            accuracies.append(row["candidate_top1_accuracy"])
        scheduler.step()
        anchors, complements = _project_catalog(model, base)
        validation = compute_complementarity_ranking_metrics(
            anchors, complements, item_ids, split.validation, subcategories
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_candidate_top1_accuracy": float(np.mean(accuracies)),
            "validation": validation,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        checkpoint = {
            "architecture": "embedding_space_rank32_residual",
            "epoch": epoch,
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout,
            "temperature": temperature,
            "model_state_dict": model.state_dict(),
            "validation": validation,
        }
        checkpoint_path = model_dir / f"epoch_{epoch:04d}.pt"
        atomic_torch_save(checkpoint, checkpoint_path)
        if epoch == selected_epoch:
            selected_checkpoint = checkpoint
            atomic_torch_save(checkpoint, save_dir / "models/comp_residual_qwen.pt")
            atomic_torch_save(
                {
                    "item_ids": item_ids,
                    "anchor_embs": anchors,
                    "complement_embs": complements,
                    "selected_epoch": selected_epoch,
                    "architecture": checkpoint["architecture"],
                },
                embedding_dir / "comp_embeddings_qwen.pt",
                expected_bytes=(anchors.numel() + complements.numel()) * 4,
            )
        atomic_json_dump(history, metrics_dir / "complementarity_history_qwen.json", indent=2)

    if selected_checkpoint is None:
        raise AssertionError("Selected residual checkpoint was not produced")
    summary = {
        "architecture": "independent embedding-space residual heads",
        "rank": rank,
        "alpha": alpha,
        "epochs_trained": epochs,
        "selected_epoch": selected_epoch,
        "selection_policy": "fixed paper checkpoint",
        "split_fingerprint": split.metadata["fingerprint"],
        "selected_validation": selected_checkpoint["validation"],
    }
    atomic_json_dump(summary, metrics_dir / "complementarity_selection_qwen.json", indent=2)
    return summary

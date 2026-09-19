from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path


try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from assembly.action_config import load_action_model_config
from assembly.action_dataset import CachedActionWindowDataset
from assembly.models.action_net import PenAssemblyActionNet
from assembly.paths import DEFAULT_ACTION_ANNOTATIONS, DEFAULT_ACTION_CONFIG, DEFAULT_FEATURE_CACHE


def _run_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, list[int], list[int]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    true_labels: list[int] = []
    predictions: list[int] = []
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(features)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * len(labels)
        true_labels.extend(labels.detach().cpu().tolist())
        predictions.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
    return total_loss / len(loader.dataset), true_labels, predictions


def main() -> int:
    parser = argparse.ArgumentParser(description="Train BiLSTM on cached ViT embeddings")
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ACTION_ANNOTATIONS)
    parser.add_argument("--config", type=Path, default=DEFAULT_ACTION_CONFIG)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "action_model")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--allow-same-session",
        action="store_true",
        help="Cho phép chia train/val/test trong cùng session (dùng khi thử nghiệm trên 1 session)",
    )
    args = parser.parse_args()

    if not args.annotations.is_file():
        template = DEFAULT_ACTION_ANNOTATIONS.parent / "annotations_SAMPLE.csv"
        raise SystemExit(
            "Chưa có file annotation cho video hành động. "
            f"Hãy sao chép {template} thành {args.annotations}, "
            "sau đó gán nhãn thời gian hoặc dùng scripts/annotate_actions.py."
        )
    if not args.features_dir.is_dir() or not any(args.features_dir.glob("*.npy")):
        raise SystemExit(
            f"Chưa có ViT feature trong {args.features_dir}. "
            "Hãy chạy scripts/extract_spatial_features.py trước."
        )

    config = load_action_model_config(args.config)
    seed = config.training.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    temporal = config.temporal

    dataset_args = dict(
        features_dir=args.features_dir,
        annotations_csv=args.annotations,
        action_to_id=config.action_to_id,
        sequence_length=temporal.sequence_length,
        stride=temporal.window_stride,
        expected_embedding_dim=config.spatial.embedding_dim,
        allow_same_session=args.allow_same_session,
    )
    train_dataset = CachedActionWindowDataset(split="train", **dataset_args)
    val_dataset = CachedActionWindowDataset(split="val", **dataset_args)
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)

    model = PenAssemblyActionNet(
        input_dim=config.spatial.embedding_dim,
        hidden_dim=temporal.hidden_dim,
        num_layers=temporal.num_layers,
        num_classes=len(config.actions),
        dropout=temporal.dropout,
        bidirectional=temporal.bidirectional,
        head_dim=temporal.head_dim,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.epochs)
    args.output.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    history: list[dict[str, float | int]] = []

    print(f"device={device} train_windows={len(train_dataset)} val_windows={len(val_dataset)}")
    for epoch in range(1, config.training.epochs + 1):
        train_loss, train_true, train_pred = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_true, val_pred = _run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        train_f1 = f1_score(train_true, train_pred, average="macro", zero_division=0)
        val_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_macro_f1": train_f1,
            "val_macro_f1": val_f1,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_f1={train_f1:.4f} val_f1={val_f1:.4f}"
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "actions": config.actions,
                    "config": asdict(config),
                    "epoch": epoch,
                    "val_macro_f1": val_f1,
                },
                args.output / "best.pt",
            )

    (args.output / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"best_val_macro_f1={best_f1:.4f} checkpoint={args.output / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


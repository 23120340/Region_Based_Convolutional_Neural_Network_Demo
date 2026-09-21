from __future__ import annotations

import argparse
import json
import sys
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

import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from assembly.action_config import load_action_model_config
from assembly.action_dataset import CachedActionWindowDataset
from assembly.models.action_net import PenAssemblyActionNet
from assembly.paths import DEFAULT_ACTION_ANNOTATIONS, DEFAULT_ACTION_CONFIG, DEFAULT_FEATURE_CACHE


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained action model on a held-out split")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ACTION_ANNOTATIONS)
    parser.add_argument("--config", type=Path, default=DEFAULT_ACTION_CONFIG)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "action_model" / "evaluation.json")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--allow-same-session",
        action="store_true",
        help="Cho phép đánh giá trên tập split cùng session",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"Không tìm thấy checkpoint: {args.checkpoint}")
    if not args.annotations.is_file():
        raise SystemExit(f"Không tìm thấy file annotation: {args.annotations}")
    if not args.features_dir.is_dir() or not any(args.features_dir.glob("*.npy")):
        raise SystemExit(f"Không tìm thấy ViT feature .npy trong: {args.features_dir}")

    config = load_action_model_config(args.config)
    temporal = config.temporal
    dataset = CachedActionWindowDataset(
        args.features_dir,
        args.annotations,
        args.split,
        config.action_to_id,
        temporal.sequence_length,
        temporal.window_stride,
        config.spatial.embedding_dim,
        allow_same_session=args.allow_same_session,
    )
    loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = PenAssemblyActionNet(
        config.spatial.embedding_dim,
        temporal.hidden_dim,
        temporal.num_layers,
        len(config.actions),
        temporal.dropout,
        temporal.bidirectional,
        temporal.head_dim,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if tuple(checkpoint.get("actions", ())) != config.actions:
        raise SystemExit("Danh sách action trong checkpoint không khớp config")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    true_labels: list[int] = []
    predictions: list[int] = []
    with torch.inference_mode():
        for features, labels in loader:
            logits = model(features.to(device))
            true_labels.extend(labels.tolist())
            predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
    label_ids = list(range(len(config.actions)))
    report = {
        "split": args.split,
        "windows": len(dataset),
        "macro_f1": f1_score(true_labels, predictions, labels=label_ids, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(true_labels, predictions, labels=label_ids).tolist(),
        "classification_report": classification_report(
            true_labels,
            predictions,
            labels=label_ids,
            target_names=list(config.actions),
            output_dict=True,
            zero_division=0,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"split": args.split, "windows": len(dataset), "macro_f1": report["macro_f1"]}, indent=2))
    print(f"Đã lưu báo cáo: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


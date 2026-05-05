from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cfc_video_demo.datasets.sequence_dataset import CfcSequenceDetectionDataset, manifest_for_split
from cfc_video_demo.models.cnn_cfc_detector import CnnCfcDetector
from cfc_video_demo.training.losses import detection_loss
from cfc_video_demo.training.metrics import DetectionMetrics
from cfc_video_demo.utils.checkpoint import save_checkpoint
from cfc_video_demo.utils.seed import seed_everything


def make_loader(ds: CfcSequenceDetectionDataset, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(
    model: CnnCfcDetector,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: str,
    box_weight: float,
    pos_weight: float | None,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "obj": 0.0, "box": 0.0}
    pbar = tqdm(loader, desc="train")
    for x, dt, obj, box in pbar:
        x, dt, obj, box = x.to(device), dt.to(device), obj.to(device), box.to(device)
        obj_logit, pred_box = model(x, dt)
        loss, parts = detection_loss(
            obj_logit,
            pred_box,
            obj,
            box,
            box_weight=box_weight,
            pos_weight=pos_weight,
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        for key in totals:
            totals[key] += parts[key]
        pbar.set_postfix(loss=f"{parts['loss']:.4f}", obj=f"{parts['obj']:.4f}", box=f"{parts['box']:.4f}")

    n = max(1, len(loader))
    return {key: value / n for key, value in totals.items()}


@torch.no_grad()
def evaluate(model: CnnCfcDetector, loader: DataLoader, device: str, threshold: float) -> dict[str, float]:
    model.eval()
    metrics = DetectionMetrics(threshold)
    for x, dt, obj, box in tqdm(loader, desc="val"):
        x, dt, obj, box = x.to(device), dt.to(device), obj.to(device), box.to(device)
        obj_logit, pred_box = model(x, dt)
        metrics.update(obj_logit, pred_box, obj, box)
    return metrics.compute()


def train(args) -> None:
    seed_everything(args.seed)
    data_root = Path(args.data)
    train_ds = CfcSequenceDetectionDataset(
        manifest_for_split(data_root, "train"),
        seq_len=args.seq_len,
        stride=args.stride,
        augment=not args.no_augment,
        max_windows=args.max_train_windows,
    )
    val_ds = CfcSequenceDetectionDataset(
        manifest_for_split(data_root, "val"),
        seq_len=args.seq_len,
        stride=args.stride,
        augment=False,
        max_windows=args.max_val_windows,
    )

    positives, total, pos_ratio = train_ds.label_stats()
    if args.pos_weight > 0:
        pos_weight = args.pos_weight
    elif positives == 0 or positives == total:
        pos_weight = 1.0
    else:
        pos_weight = (total - positives) / max(1, positives)
    print(f"Train windows: {len(train_ds)}")
    print(f"Val windows: {len(val_ds)}")
    print(f"Train positive windows: {positives}/{total} ({pos_ratio:.1%})")
    print(f"BCE pos_weight: {pos_weight:.3f}")

    train_loader = make_loader(train_ds, args.batch, args.workers, shuffle=True)
    val_loader = make_loader(val_ds, args.batch, args.workers, shuffle=False)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model = CnnCfcDetector(args.image_size, args.feat_dim, args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = -1.0
    best_f1 = -1.0
    best_metrics: dict[str, float] = {}
    for epoch in range(1, args.epochs + 1):
        losses = train_one_epoch(model, train_loader, opt, device, args.box_weight, pos_weight)
        metrics = evaluate(model, val_loader, device, args.th)
        print(
            f"epoch={epoch} loss={losses['loss']:.4f} obj={losses['obj']:.4f} box={losses['box']:.4f} "
            f"precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} "
            f"f1={metrics['f1']:.3f} mean_iou={metrics['mean_iou']:.3f} "
            f"recall_at_iou_0_5={metrics['recall_at_iou_0_5']:.3f}"
        )

        score = metrics["recall_at_iou_0_5"]
        if score > best_score or (score == best_score and metrics["f1"] >= best_f1):
            best_score = score
            best_f1 = metrics["f1"]
            best_metrics = metrics
            save_checkpoint(
                args.model,
                {
                    "model": model.state_dict(),
                    "config": {
                        "image_size": args.image_size,
                        "seq_len": args.seq_len,
                        "stride": args.stride,
                        "feat_dim": args.feat_dim,
                        "hidden": args.hidden,
                        "arch": "cnn-cfc-single-target-detector-v1",
                    },
                    "epoch": epoch,
                    "metrics": metrics,
                    "label_format": "single target objectness + normalized cx cy w h",
                },
            )
            print(f"Saved best checkpoint: {args.model}")

    print(f"Best metrics: {best_metrics}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CNN encoder + CfC single-target pedestrian detector.")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--box-weight", type=float, default=10.0)
    parser.add_argument("--pos-weight", type=float, default=0.0)
    parser.add_argument("--feat-dim", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--th", type=float, default=0.5)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-val-windows", type=int, default=0)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.seq_len < 2:
        raise ValueError("--seq-len must be >= 2")
    train(args)


if __name__ == "__main__":
    main()

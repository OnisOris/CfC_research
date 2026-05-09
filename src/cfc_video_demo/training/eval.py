from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cfc_video_demo.datasets.sequence_dataset import CfcSequenceDetectionDataset, manifest_for_split
from cfc_video_demo.models.cnn_cfc_detector import CnnCfcDetector
from cfc_video_demo.training.metrics import DetectionMetrics
from cfc_video_demo.utils.checkpoint import load_checkpoint


@torch.no_grad()
def run_eval(args) -> dict[str, float]:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    ckpt = load_checkpoint(args.model, device)
    config = ckpt["config"]
    seq_len = args.seq_len if args.seq_len > 0 else int(config["seq_len"])
    stride = args.stride if args.stride > 0 else int(config.get("stride", 4))

    ds = CfcSequenceDetectionDataset(
        manifest_for_split(Path(args.data), args.split),
        seq_len=seq_len,
        stride=stride,
        augment=False,
        max_windows=args.max_windows,
    )
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    model = CnnCfcDetector(
        image_size=int(config.get("image_size", 128)),
        feat_dim=int(config.get("feat_dim", 128)),
        hidden=int(config.get("hidden", 128)),
        spatial_pool=int(config.get("spatial_pool", 4)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    metrics = DetectionMetrics(args.th)
    for x, dt, obj, box in tqdm(loader, desc=f"eval:{args.split}"):
        x, dt, obj, box = x.to(device), dt.to(device), obj.to(device), box.to(device)
        obj_logit, pred_box = model(x, dt)
        metrics.update(obj_logit, pred_box, obj, box)
    return metrics.compute()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a CNN+CfC Caltech checkpoint.")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=("train", "val", "test"))
    parser.add_argument("--th", type=float, default=0.5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=0)
    parser.add_argument("--stride", type=int, default=0)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    metrics = run_eval(args)
    for key in ("precision", "recall", "f1", "mean_iou", "recall_at_iou_0_5"):
        print(f"{key}: {metrics[key]:.4f}")


if __name__ == "__main__":
    main()

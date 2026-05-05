from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch


def clip_xyxy(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[float, float, float, float]:
    x1 = float(np.clip(x1, 0, width))
    x2 = float(np.clip(x2, 0, width))
    y1 = float(np.clip(y1, 0, height))
    y2 = float(np.clip(y2, 0, height))
    return x1, y1, x2, y2


def xywh_to_cxcywh_norm(box: Iterable[float], width: int, height: int) -> np.ndarray:
    x, y, w, h = [float(v) for v in box]
    x1, y1, x2, y2 = clip_xyxy(x, y, x + w, y + h, width, height)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return np.zeros(4, dtype=np.float32)
    return np.array(
        [
            ((x1 + x2) / 2.0) / width,
            ((y1 + y2) / 2.0) / height,
            (x2 - x1) / width,
            (y2 - y1) / height,
        ],
        dtype=np.float32,
    )


def choose_single_target(
    boxes_xywh: Iterable[Iterable[float]],
    width: int,
    height: int,
    mode: str = "largest",
    min_box_height: float = 10.0,
) -> tuple[float, np.ndarray]:
    boxes = []
    for raw in boxes_xywh:
        x, y, w, h = [float(v) for v in raw]
        if w <= 1 or h < min_box_height:
            continue
        x1, y1, x2, y2 = clip_xyxy(x, y, x + w, y + h, width, height)
        if x2 - x1 < 1 or y2 - y1 < min_box_height:
            continue
        boxes.append((x1, y1, x2 - x1, y2 - y1))

    if not boxes:
        return 0.0, np.zeros(4, dtype=np.float32)

    if mode == "largest":
        box = max(boxes, key=lambda b: b[2] * b[3])
        return 1.0, xywh_to_cxcywh_norm(box, width, height)

    if mode == "union":
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[0] + b[2] for b in boxes)
        y2 = max(b[1] + b[3] for b in boxes)
        return 1.0, xywh_to_cxcywh_norm((x1, y1, x2 - x1, y2 - y1), width, height)

    raise ValueError(f"Unknown target mode: {mode}")


def cxcywh_to_xyxy_np(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    cx, cy, bw, bh = [float(v) for v in box]
    x1 = int(round((cx - bw / 2.0) * width))
    y1 = int(round((cy - bh / 2.0) * height))
    x2 = int(round((cx + bw / 2.0) * width))
    y2 = int(round((cy + bh / 2.0) * height))
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    return x1, y1, x2, y2


def box_iou_cxcywh_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_x1 = pred[:, 0] - pred[:, 2] / 2.0
    pred_y1 = pred[:, 1] - pred[:, 3] / 2.0
    pred_x2 = pred[:, 0] + pred[:, 2] / 2.0
    pred_y2 = pred[:, 1] + pred[:, 3] / 2.0

    tgt_x1 = target[:, 0] - target[:, 2] / 2.0
    tgt_y1 = target[:, 1] - target[:, 3] / 2.0
    tgt_x2 = target[:, 0] + target[:, 2] / 2.0
    tgt_y2 = target[:, 1] + target[:, 3] / 2.0

    ix1 = torch.maximum(pred_x1, tgt_x1)
    iy1 = torch.maximum(pred_y1, tgt_y1)
    ix2 = torch.minimum(pred_x2, tgt_x2)
    iy2 = torch.minimum(pred_y2, tgt_y2)

    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    pred_area = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
    tgt_area = (tgt_x2 - tgt_x1).clamp(min=0) * (tgt_y2 - tgt_y1).clamp(min=0)
    return inter / (pred_area + tgt_area - inter + 1e-7)


def box_iou_cxcywh_np(pred: np.ndarray, target: np.ndarray) -> float:
    pred_t = torch.tensor(np.asarray(pred, dtype=np.float32)).reshape(1, 4)
    target_t = torch.tensor(np.asarray(target, dtype=np.float32)).reshape(1, 4)
    return float(box_iou_cxcywh_torch(pred_t, target_t)[0].item())


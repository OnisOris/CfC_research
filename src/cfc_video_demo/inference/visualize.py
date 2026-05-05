from __future__ import annotations

import cv2
import numpy as np

from cfc_video_demo.utils.boxes import cxcywh_to_xyxy_np


def draw_box(
    frame_bgr: np.ndarray,
    box: np.ndarray,
    color: tuple[int, int, int],
    label: str | None = None,
    thickness: int = 2,
) -> None:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = cxcywh_to_xyxy_np(box, w, h)
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(frame_bgr, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_status(frame_bgr: np.ndarray, prob: float | None, iou: float | None) -> None:
    if prob is None:
        text = "CfC person prob: warming up"
    else:
        text = f"CfC person prob: {prob:.2f}"
    cv2.putText(frame_bgr, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    if iou is not None:
        cv2.putText(frame_bgr, f"IoU: {iou:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)


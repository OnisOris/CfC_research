from __future__ import annotations

import torch

from cfc_video_demo.utils.boxes import box_iou_cxcywh_torch


class DetectionMetrics:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.positive_targets = 0
        self.iou_sum = 0.0
        self.iou_count = 0
        self.recall_iou_hits = 0

    @torch.no_grad()
    def update(
        self,
        obj_logit: torch.Tensor,
        pred_box: torch.Tensor,
        obj: torch.Tensor,
        target_box: torch.Tensor,
    ) -> None:
        prob = torch.sigmoid(obj_logit)
        pred_obj = prob > self.threshold
        target_obj = obj > 0.5

        self.tp += int((pred_obj & target_obj).sum().item())
        self.fp += int((pred_obj & ~target_obj).sum().item())
        self.fn += int((~pred_obj & target_obj).sum().item())

        if target_obj.any():
            iou = box_iou_cxcywh_torch(pred_box[target_obj], target_box[target_obj])
            self.positive_targets += int(target_obj.sum().item())
            self.iou_sum += float(iou.sum().item())
            self.iou_count += int(iou.numel())
            pred_positive_iou = pred_obj[target_obj] & (iou >= 0.5)
            self.recall_iou_hits += int(pred_positive_iou.sum().item())

    def compute(self) -> dict[str, float]:
        precision = self.tp / max(1, self.tp + self.fp)
        recall = self.tp / max(1, self.tp + self.fn)
        f1 = 2 * precision * recall / max(1e-7, precision + recall)
        mean_iou = self.iou_sum / max(1, self.iou_count)
        recall_at_iou = self.recall_iou_hits / max(1, self.positive_targets)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_iou": mean_iou,
            "recall_at_iou_0_5": recall_at_iou,
        }


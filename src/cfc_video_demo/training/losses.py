from __future__ import annotations

import torch
import torch.nn.functional as F


def detection_loss(
    obj_logit: torch.Tensor,
    pred_box: torch.Tensor,
    obj: torch.Tensor,
    target_box: torch.Tensor,
    box_weight: float = 10.0,
    pos_weight: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if pos_weight is not None:
        pos_weight_t = torch.as_tensor(pos_weight, device=obj.device, dtype=obj.dtype)
        obj_loss = F.binary_cross_entropy_with_logits(obj_logit, obj, pos_weight=pos_weight_t)
    else:
        obj_loss = F.binary_cross_entropy_with_logits(obj_logit, obj)

    per_box = F.smooth_l1_loss(pred_box, target_box, reduction="none").mean(dim=1)
    positive = obj > 0.5
    if positive.any():
        box_loss = per_box[positive].mean()
    else:
        box_loss = pred_box.sum() * 0.0

    total = obj_loss + box_weight * box_loss
    return total, {
        "loss": float(total.detach().cpu().item()),
        "obj": float(obj_loss.detach().cpu().item()),
        "box": float(box_loss.detach().cpu().item()),
    }


from __future__ import annotations

import torch
import torch.nn.functional as F


def _cxcywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = box.unbind(dim=-1)
    return torch.stack(
        (
            cx - w / 2.0,
            cy - h / 2.0,
            cx + w / 2.0,
            cy + h / 2.0,
        ),
        dim=-1,
    )


def distance_iou_loss(pred_box: torch.Tensor, target_box: torch.Tensor) -> torch.Tensor:
    pred = _cxcywh_to_xyxy(pred_box)
    target = _cxcywh_to_xyxy(target_box)

    inter_x1 = torch.maximum(pred[:, 0], target[:, 0])
    inter_y1 = torch.maximum(pred[:, 1], target[:, 1])
    inter_x2 = torch.minimum(pred[:, 2], target[:, 2])
    inter_y2 = torch.minimum(pred[:, 3], target[:, 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    pred_area = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
    target_area = (target[:, 2] - target[:, 0]).clamp(min=0) * (target[:, 3] - target[:, 1]).clamp(min=0)
    iou = inter / (pred_area + target_area - inter + 1e-7)

    center_dist = (pred_box[:, 0] - target_box[:, 0]).square() + (pred_box[:, 1] - target_box[:, 1]).square()
    enc_x1 = torch.minimum(pred[:, 0], target[:, 0])
    enc_y1 = torch.minimum(pred[:, 1], target[:, 1])
    enc_x2 = torch.maximum(pred[:, 2], target[:, 2])
    enc_y2 = torch.maximum(pred[:, 3], target[:, 3])
    enc_diag = (enc_x2 - enc_x1).square() + (enc_y2 - enc_y1).square() + 1e-7
    return (1.0 - iou + center_dist / enc_diag).mean()


def detection_loss(
    obj_logit: torch.Tensor,
    pred_box: torch.Tensor,
    obj: torch.Tensor,
    target_box: torch.Tensor,
    box_weight: float = 10.0,
    iou_weight: float = 0.0,
    heatmap_logits: torch.Tensor | None = None,
    heatmap_weight: float = 0.0,
    cell_boxes: torch.Tensor | None = None,
    grid_size: torch.Tensor | tuple[int, int] | None = None,
    pos_weight: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if pos_weight is not None:
        pos_weight_t = torch.as_tensor(pos_weight, device=obj.device, dtype=obj.dtype)
        obj_loss = F.binary_cross_entropy_with_logits(obj_logit, obj, pos_weight=pos_weight_t)
    else:
        obj_loss = F.binary_cross_entropy_with_logits(obj_logit, obj)

    positive = obj > 0.5
    target_cell = None
    if heatmap_logits is not None:
        if grid_size is None:
            cells = heatmap_logits.shape[1]
            grid_h = grid_w = int(cells**0.5)
            if grid_h * grid_w != cells:
                raise ValueError(f"heatmap cells must form a square grid, got {cells}")
        else:
            if torch.is_tensor(grid_size):
                grid_h = int(grid_size[0].detach().cpu().item())
                grid_w = int(grid_size[1].detach().cpu().item())
            else:
                grid_h = int(grid_size[0])
                grid_w = int(grid_size[1])

        cx = target_box[:, 0].clamp(0, 1 - 1e-6)
        cy = target_box[:, 1].clamp(0, 1 - 1e-6)
        target_x = torch.floor(cx * grid_w).long()
        target_y = torch.floor(cy * grid_h).long()
        target_cell = target_y * grid_w + target_x

    box_source = pred_box
    if cell_boxes is not None and target_cell is not None:
        box_source = cell_boxes[torch.arange(cell_boxes.shape[0], device=cell_boxes.device), target_cell]

    per_box = F.smooth_l1_loss(box_source, target_box, reduction="none").mean(dim=1)
    if positive.any():
        box_loss = per_box[positive].mean()
        iou_loss = distance_iou_loss(box_source[positive], target_box[positive])
    else:
        box_loss = pred_box.sum() * 0.0
        iou_loss = pred_box.sum() * 0.0

    if heatmap_logits is not None and heatmap_weight > 0 and target_cell is not None:
        target_heatmap = torch.zeros_like(heatmap_logits)
        if positive.any():
            target_heatmap[positive, target_cell[positive]] = 1.0
        heatmap_parts = F.binary_cross_entropy_with_logits(heatmap_logits, target_heatmap, reduction="none")
        if positive.any():
            weights = torch.ones_like(heatmap_parts)
            weights[positive, target_cell[positive]] = heatmap_logits.shape[1]
            heatmap_loss = (heatmap_parts * weights).mean()
        else:
            heatmap_loss = heatmap_parts.mean()
    else:
        heatmap_loss = pred_box.sum() * 0.0

    total = obj_loss + box_weight * (box_loss + iou_weight * iou_loss) + heatmap_weight * heatmap_loss
    return total, {
        "loss": float(total.detach().cpu().item()),
        "obj": float(obj_loss.detach().cpu().item()),
        "box": float(box_loss.detach().cpu().item()),
        "iou": float(iou_loss.detach().cpu().item()),
        "heatmap": float(heatmap_loss.detach().cpu().item()),
    }

from __future__ import annotations

import torch
import torch.nn as nn
from ncps.torch import CfC


class CnnCfcDetector(nn.Module):
    def __init__(
        self,
        image_size: int = 128,
        feat_dim: int = 128,
        hidden: int = 128,
        spatial_pool: int = 4,
    ):
        super().__init__()
        self.image_size = image_size
        self.feat_dim = feat_dim
        self.hidden = hidden
        self.spatial_pool = spatial_pool
        self._coord_cache: dict[tuple[torch.device, torch.dtype, int, int], torch.Tensor] = {}

        self.encoder = nn.Sequential(
            nn.Conv2d(5, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(),
            nn.Conv2d(96, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 160, 3, stride=2, padding=1),
            nn.BatchNorm2d(160),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((spatial_pool, spatial_pool)),
        )
        self.frame_proj = nn.Sequential(
            nn.Conv2d(160, feat_dim, 1),
            nn.SiLU(),
        )
        self.cfc = CfC(input_size=feat_dim + 2, units=hidden)
        self.cell_head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.SiLU(),
            nn.Linear(128, 5),
        )
        self._grid_cache: dict[tuple[torch.device, torch.dtype, int, int], torch.Tensor] = {}

    def add_coord_channels(self, x: torch.Tensor) -> torch.Tensor:
        b, _c, h, w = x.shape
        key = (x.device, x.dtype, h, w)
        coords = self._coord_cache.get(key)
        if coords is None:
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype),
                torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype),
                indexing="ij",
            )
            coords = torch.stack([xx, yy], dim=0).unsqueeze(0)
            self._coord_cache[key] = coords
        coords = coords.expand(b, -1, -1, -1)
        return torch.cat([x, coords], dim=1)

    def grid_centers(self, device: torch.device, dtype: torch.dtype, h: int, w: int) -> torch.Tensor:
        key = (device, dtype, h, w)
        centers = self._grid_cache.get(key)
        if centers is None:
            yy, xx = torch.meshgrid(
                (torch.arange(h, device=device, dtype=dtype) + 0.5) / h,
                (torch.arange(w, device=device, dtype=dtype) + 0.5) / w,
                indexing="ij",
            )
            centers = torch.stack([xx, yy], dim=-1).reshape(h * w, 2)
            self._grid_cache[key] = centers
        return centers

    def forward(
        self,
        x: torch.Tensor,
        dt: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        z = self.frame_proj(self.encoder(self.add_coord_channels(x)))
        _bt, feat_dim, gh, gw = z.shape
        cells = gh * gw

        z = z.reshape(b, t, feat_dim, cells).permute(0, 3, 1, 2)
        centers = self.grid_centers(z.device, z.dtype, gh, gw)
        center_features = centers.unsqueeze(0).unsqueeze(2).expand(b, cells, t, 2)
        z = torch.cat([z, center_features], dim=-1).reshape(b * cells, t, feat_dim + 2)

        if dt is not None and dt.ndim == 2:
            dt = dt.unsqueeze(1).expand(-1, cells, -1).reshape(b * cells, t)
            dt = dt.unsqueeze(-1).expand(-1, -1, self.hidden)

        try:
            y, _hn = self.cfc(z, timespans=dt)
        except TypeError:
            y, _hn = self.cfc(z)

        if y.ndim == 3:
            y = y[:, -1, :]

        out = self.cell_head(y).reshape(b, cells, 5)
        cell_logits = out[..., 0]
        cell_offset = torch.sigmoid(out[..., 1:3])
        cell_wh = torch.sigmoid(out[..., 3:5])

        cell_size = centers.new_tensor([1.0 / gw, 1.0 / gh])
        cell_origin = centers - 0.5 * cell_size
        cell_center = cell_origin.unsqueeze(0) + cell_offset * cell_size
        cell_boxes = torch.cat([cell_center, cell_wh], dim=-1)

        best_cell = cell_logits.argmax(dim=1)
        gather_idx = best_cell.view(b, 1, 1).expand(-1, 1, 4)
        box = cell_boxes.gather(dim=1, index=gather_idx).squeeze(1)
        obj_logit = cell_logits.gather(dim=1, index=best_cell.view(b, 1)).squeeze(1)

        if return_aux:
            aux = {
                "heatmap_logits": cell_logits,
                "cell_logits": cell_logits,
                "cell_boxes": cell_boxes,
                "grid_size": torch.tensor([gh, gw], device=x.device),
            }
            return obj_logit, box, aux
        return obj_logit, box


class YoloCfcRefiner(nn.Module):
    def __init__(
        self,
        input_size: int = 5,
        hidden: int = 96,
        direct_weight: float = 0.25,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden = hidden
        self.direct_weight = direct_weight
        self.cfc = CfC(input_size=input_size, units=hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.SiLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor, dt: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if dt is not None and dt.ndim == 2:
            dt = dt.unsqueeze(-1).expand(-1, -1, self.hidden)

        try:
            y, _hn = self.cfc(x, timespans=dt)
        except TypeError:
            y, _hn = self.cfc(x)

        if y.ndim == 3:
            y = y[:, -1, :]

        out = self.head(y)
        last_conf = x[:, -1, 0].clamp(1e-4, 1.0 - 1e-4)
        base_box = x[:, -1, 1:5]
        base_logit = torch.logit(last_conf)

        obj_logit = out[:, 0] + base_logit
        residual = torch.tanh(out[:, 1:5]) * x.new_tensor([0.25, 0.25, 0.5, 0.5])
        residual_box = (base_box + residual).clamp(0.0, 1.0)
        direct_box = torch.sigmoid(out[:, 5:9])
        mix = torch.sigmoid(out[:, 9:10])
        box = (
            mix * residual_box
            + (1.0 - mix) * self.direct_weight * direct_box
            + (1.0 - mix) * (1.0 - self.direct_weight) * residual_box
        )
        return obj_logit, box.clamp(0.0, 1.0)

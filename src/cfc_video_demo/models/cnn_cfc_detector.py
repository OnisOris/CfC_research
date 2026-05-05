from __future__ import annotations

import torch
import torch.nn as nn
from ncps.torch import CfC


class CnnCfcDetector(nn.Module):
    def __init__(self, image_size: int = 128, feat_dim: int = 128, hidden: int = 128):
        super().__init__()
        self.image_size = image_size
        self.feat_dim = feat_dim
        self.hidden = hidden

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
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(160 * 4 * 4, feat_dim),
            nn.SiLU(),
        )
        self.cfc = CfC(input_size=feat_dim, units=hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.SiLU(),
            nn.Linear(128, 5),
        )

    def add_coord_channels(self, x: torch.Tensor) -> torch.Tensor:
        b, _c, h, w = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(b, -1, -1, -1)
        return torch.cat([x, coords], dim=1)

    def forward(self, x: torch.Tensor, dt: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        z = self.encoder(self.add_coord_channels(x))
        z = z.reshape(b, t, -1)

        if dt is not None and dt.ndim == 2:
            dt = dt.unsqueeze(-1).expand(-1, -1, self.hidden)

        try:
            y, _hn = self.cfc(z, timespans=dt)
        except TypeError:
            y, _hn = self.cfc(z)

        if y.ndim == 3:
            y = y[:, -1, :]

        out = self.head(y)
        obj_logit = out[:, 0]
        box = torch.sigmoid(out[:, 1:5])
        return obj_logit, box


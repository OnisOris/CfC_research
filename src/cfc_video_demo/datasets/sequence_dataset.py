from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ManifestEntry:
    path: Path
    sequence: str
    split: str
    frames: int
    positive_frames: int
    fps: float


class CfcSequenceDetectionDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        seq_len: int = 16,
        stride: int = 4,
        augment: bool = False,
        max_windows: int = 0,
        cache_size: int = 2,
    ):
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.seq_len = seq_len
        self.stride = stride
        self.augment = augment
        self.cache_size = cache_size
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

        if seq_len < 2:
            raise ValueError("seq_len must be >= 2 for a temporal CfC dataset")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {self.manifest_path}")

        self.entries = self._read_manifest(self.manifest_path)
        if not self.entries:
            raise RuntimeError(f"Manifest is empty: {self.manifest_path}")

        self.index: list[tuple[int, int]] = []
        for entry_idx, entry in enumerate(self.entries):
            if entry.frames < seq_len:
                continue
            for start in range(0, entry.frames - seq_len + 1, stride):
                self.index.append((entry_idx, start))

        if max_windows > 0:
            self.index = self.index[:max_windows]
        if not self.index:
            raise RuntimeError(
                f"No windows in {self.manifest_path}; frames are shorter than seq_len={seq_len}"
            )

    def _read_manifest(self, path: Path) -> list[ManifestEntry]:
        entries = []
        for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
            if not raw.strip():
                continue
            item = json.loads(raw)
            rel_path = Path(item["path"])
            npz_path = rel_path if rel_path.is_absolute() else self.root / rel_path
            entries.append(
                ManifestEntry(
                    path=npz_path,
                    sequence=str(item.get("sequence", npz_path.stem)),
                    split=str(item.get("split", path.stem.replace("manifest_", ""))),
                    frames=int(item["frames"]),
                    positive_frames=int(item.get("positive_frames", 0)),
                    fps=float(item.get("fps", 30.0)),
                )
            )
            if not npz_path.exists():
                raise FileNotFoundError(f"Manifest line {lineno} points to missing file: {npz_path}")
        return entries

    def __len__(self) -> int:
        return len(self.index)

    def _load_entry(self, entry_idx: int) -> dict[str, np.ndarray]:
        if entry_idx in self._cache:
            data = self._cache.pop(entry_idx)
            self._cache[entry_idx] = data
            return data

        entry = self.entries[entry_idx]
        npz = np.load(entry.path, allow_pickle=False)
        data = {
            "frames": npz["frames"],
            "times": npz["times"].astype(np.float32),
            "labels_obj": npz["labels_obj"].astype(np.float32),
            "labels_box": npz["labels_box"].astype(np.float32),
        }
        self._cache[entry_idx] = data
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return data

    def __getitem__(self, idx: int):
        entry_idx, start = self.index[idx]
        end = start + self.seq_len
        data = self._load_entry(entry_idx)

        frames = data["frames"][start:end].astype(np.float32) / 255.0
        times = data["times"][start:end].astype(np.float32)
        obj = np.float32(data["labels_obj"][end - 1])
        box = data["labels_box"][end - 1].astype(np.float32).copy()

        if self.augment and np.random.random() < 0.5:
            frames = frames[:, :, ::-1, :].copy()
            if obj > 0:
                box[0] = 1.0 - box[0]

        dt = np.diff(times, prepend=times[0])
        if len(dt) > 1:
            valid = dt[1:][dt[1:] > 1e-4]
            dt[0] = np.median(valid) if len(valid) else 1.0 / 30.0
        dt = np.clip(dt, 1e-4, 1.0).astype(np.float32)

        x = frames.transpose(0, 3, 1, 2)
        return (
            torch.from_numpy(x),
            torch.from_numpy(dt),
            torch.tensor(obj, dtype=torch.float32),
            torch.from_numpy(box),
        )

    def label_stats(self) -> tuple[int, int, float]:
        positives = 0
        for entry_idx, start in self.index:
            data = self._load_entry(entry_idx)
            target_idx = start + self.seq_len - 1
            positives += int(data["labels_obj"][target_idx] > 0.5)
        total = len(self.index)
        ratio = positives / total if total else 0.0
        return positives, total, ratio


class CfcYoloFeatureDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        seq_len: int = 16,
        stride: int = 4,
        augment: bool = False,
        max_windows: int = 0,
        cache_size: int = 16,
    ):
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.seq_len = seq_len
        self.stride = stride
        self.augment = augment
        self.cache_size = cache_size
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

        if seq_len < 2:
            raise ValueError("seq_len must be >= 2 for a temporal CfC dataset")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {self.manifest_path}")

        self.entries = CfcSequenceDetectionDataset._read_manifest(self, self.manifest_path)
        if not self.entries:
            raise RuntimeError(f"Manifest is empty: {self.manifest_path}")

        self.index: list[tuple[int, int]] = []
        for entry_idx, entry in enumerate(self.entries):
            if entry.frames < seq_len:
                continue
            for start in range(0, entry.frames - seq_len + 1, stride):
                self.index.append((entry_idx, start))

        if max_windows > 0:
            self.index = self.index[:max_windows]
        if not self.index:
            raise RuntimeError(
                f"No windows in {self.manifest_path}; frames are shorter than seq_len={seq_len}"
            )

    def __len__(self) -> int:
        return len(self.index)

    def _load_entry(self, entry_idx: int) -> dict[str, np.ndarray]:
        if entry_idx in self._cache:
            data = self._cache.pop(entry_idx)
            self._cache[entry_idx] = data
            return data

        entry = self.entries[entry_idx]
        npz = np.load(entry.path, allow_pickle=False)
        if "yolo_features" not in npz:
            raise RuntimeError(
                f"{entry.path} has no yolo_features. Run cfc-yolo-cache before cfc-yolo-train."
            )
        features = npz["yolo_features"].astype(np.float32)
        if features.ndim != 2 or features.shape[1] != 5:
            raise RuntimeError(f"{entry.path} yolo_features must have shape [frames, 5], got {features.shape}")
        data = {
            "features": features,
            "times": npz["times"].astype(np.float32),
            "labels_obj": npz["labels_obj"].astype(np.float32),
            "labels_box": npz["labels_box"].astype(np.float32),
        }
        self._cache[entry_idx] = data
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return data

    def __getitem__(self, idx: int):
        entry_idx, start = self.index[idx]
        end = start + self.seq_len
        data = self._load_entry(entry_idx)

        features = data["features"][start:end].astype(np.float32).copy()
        times = data["times"][start:end].astype(np.float32)
        obj = np.float32(data["labels_obj"][end - 1])
        box = data["labels_box"][end - 1].astype(np.float32).copy()

        if self.augment and np.random.random() < 0.5:
            features[:, 1] = 1.0 - features[:, 1]
            if obj > 0:
                box[0] = 1.0 - box[0]

        dt = np.diff(times, prepend=times[0])
        if len(dt) > 1:
            valid = dt[1:][dt[1:] > 1e-4]
            dt[0] = np.median(valid) if len(valid) else 1.0 / 30.0
        dt = np.clip(dt, 1e-4, 1.0).astype(np.float32)

        return (
            torch.from_numpy(features),
            torch.from_numpy(dt),
            torch.tensor(obj, dtype=torch.float32),
            torch.from_numpy(box),
        )

    def label_stats(self) -> tuple[int, int, float]:
        positives = 0
        for entry_idx, start in self.index:
            data = self._load_entry(entry_idx)
            target_idx = start + self.seq_len - 1
            positives += int(data["labels_obj"][target_idx] > 0.5)
        total = len(self.index)
        ratio = positives / total if total else 0.0
        return positives, total, ratio


def manifest_for_split(data_root: str | Path, split: str) -> Path:
    root = Path(data_root)
    path = root / f"manifest_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing split manifest: {path}")
    return path

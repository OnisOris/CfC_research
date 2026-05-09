from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_yolo(model_name: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Missing ultralytics. Install dependencies with `uv sync` first.") from exc
    return YOLO(model_name)


def top_person_feature(result) -> np.ndarray:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return np.zeros(5, dtype=np.float32)

    conf = boxes.conf.detach().cpu().numpy()
    xyxy = boxes.xyxy.detach().cpu().numpy()
    idx = int(conf.argmax())
    x1, y1, x2, y2 = [float(v) for v in xyxy[idx]]
    height, width = result.orig_shape[:2]
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return np.array(
        [
            float(conf[idx]),
            np.clip(cx, 0.0, 1.0),
            np.clip(cy, 0.0, 1.0),
            np.clip(bw, 0.0, 1.0),
            np.clip(bh, 0.0, 1.0),
        ],
        dtype=np.float32,
    )


def iter_manifest_paths(data_root: Path, splits: list[str]) -> list[Path]:
    paths: list[Path] = []
    for split in splits:
        manifest = data_root / f"manifest_{split}.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest}")
        for raw in manifest.read_text().splitlines():
            if not raw.strip():
                continue
            item = json.loads(raw)
            rel_path = Path(item["path"])
            paths.append(rel_path if rel_path.is_absolute() else data_root / rel_path)
    return paths


def rewrite_npz(path: Path, updates: dict[str, np.ndarray]) -> None:
    with np.load(path, allow_pickle=False) as npz:
        payload = {key: npz[key] for key in npz.files}
    payload.update(updates)
    np.savez_compressed(path, **payload)


def cache_sequence(model, path: Path, args) -> None:
    with np.load(path, allow_pickle=False) as npz:
        if "yolo_features" in npz and not args.overwrite:
            return
        frames = npz["frames"]

    features = np.zeros((len(frames), 5), dtype=np.float32)
    for start in tqdm(range(0, len(frames), args.batch), desc=path.stem, leave=False):
        batch = [frame for frame in frames[start : start + args.batch]]
        results = model.predict(
            source=batch,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            classes=[0],
            device=args.device or None,
            verbose=False,
        )
        for offset, result in enumerate(results):
            features[start + offset] = top_person_feature(result)

    rewrite_npz(
        path,
        {
            "yolo_features": features,
            "yolo_model": np.array(args.yolo_model),
            "yolo_feature_format": np.array("conf cx cy w h"),
        },
    )


def run_cache(args) -> None:
    data_root = Path(args.data)
    model = load_yolo(args.yolo_model)
    splits = args.splits.split(",")
    paths = iter_manifest_paths(data_root, splits)
    if args.max_sequences > 0:
        paths = paths[: args.max_sequences]
    print(f"Sequences: {len(paths)}")
    for path in tqdm(paths, desc="cache"):
        cache_sequence(model, path, args)
    print(f"Saved YOLO feature cache into: {data_root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache pretrained YOLO person detections into prepared Caltech npz files.")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt")
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    run_cache(parser.parse_args())


if __name__ == "__main__":
    main()

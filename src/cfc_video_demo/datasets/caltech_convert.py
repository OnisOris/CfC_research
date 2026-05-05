from __future__ import annotations

import argparse
import json
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from cfc_video_demo.datasets.caltech import (
    find_set_name,
    image_index,
    iter_seq_frames,
    read_json_boxes,
    read_seq_header,
    read_vbb_boxes,
    sequence_name,
    sorted_image_paths,
)
from cfc_video_demo.utils.boxes import choose_single_target
from cfc_video_demo.utils.video import resize_rgb


TRAIN_SETS = {f"set{i:02d}" for i in range(0, 6)}
TEST_SETS = {f"set{i:02d}" for i in range(6, 11)}


@dataclass(frozen=True)
class RawSequence:
    set_name: str
    video_name: str
    seq_path: Path | None = None
    vbb_path: Path | None = None
    images_dir: Path | None = None
    annotations_dir: Path | None = None

    @property
    def name(self) -> str:
        return sequence_name(self.set_name, self.video_name)


@dataclass
class SequenceStats:
    frames: int = 0
    positive_frames: int = 0
    box_width_sum: float = 0.0
    box_height_sum: float = 0.0

    def add(self, obj: float, box: np.ndarray) -> None:
        self.frames += 1
        if obj > 0:
            self.positive_frames += 1
            self.box_width_sum += float(box[2])
            self.box_height_sum += float(box[3])

    def merge(self, other: "SequenceStats") -> None:
        self.frames += other.frames
        self.positive_frames += other.positive_frames
        self.box_width_sum += other.box_width_sum
        self.box_height_sum += other.box_height_sum


def discover_preextracted(raw_root: Path) -> list[RawSequence]:
    sequences = []
    for images_dir in sorted(raw_root.rglob("images")):
        if not images_dir.is_dir() or not sorted_image_paths(images_dir):
            continue
        annotations_dir = images_dir.parent / "annotations"
        if not annotations_dir.is_dir():
            continue
        set_name = find_set_name(images_dir)
        if not set_name:
            continue
        sequences.append(
            RawSequence(
                set_name=set_name,
                video_name=images_dir.parent.name,
                images_dir=images_dir,
                annotations_dir=annotations_dir,
            )
        )
    return sorted(sequences, key=lambda s: (s.set_name, s.video_name))


def discover_seq_vbb(raw_root: Path) -> list[RawSequence]:
    vbb_index: dict[tuple[str, str], Path] = {}
    for vbb_path in raw_root.rglob("*.vbb"):
        set_name = find_set_name(vbb_path)
        if set_name:
            vbb_index[(set_name, vbb_path.stem)] = vbb_path

    sequences = []
    for seq_path in sorted(raw_root.rglob("*.seq")):
        set_name = find_set_name(seq_path)
        if not set_name:
            continue
        vbb_path = vbb_index.get((set_name, seq_path.stem))
        if vbb_path is None:
            continue
        sequences.append(
            RawSequence(
                set_name=set_name,
                video_name=seq_path.stem,
                seq_path=seq_path,
                vbb_path=vbb_path,
            )
        )
    return sorted(sequences, key=lambda s: (s.set_name, s.video_name))


def maybe_print_archive_hint(raw_root: Path) -> None:
    archives = sorted([*raw_root.rglob("*.tar"), *raw_root.rglob("*.zip")])
    if archives:
        print("Found archives but no usable Caltech sequences were discovered yet.")
        print("Unpack the Caltech setXX.tar files and annotations.zip under --raw-root.")
        for path in archives[:8]:
            print(f"  archive: {path}")


def ensure_safe_extract_path(out_dir: Path, member_name: str) -> None:
    root = out_dir.resolve()
    target = (out_dir / member_name).resolve()
    if root != target and root not in target.parents:
        raise RuntimeError(f"Unsafe archive member path: {member_name}")


def extract_archives(raw_root: Path) -> None:
    archives = sorted(raw_root.rglob("*.tar"))
    for archive in archives:
        out_dir = archive.parent
        marker = out_dir / f".{archive.stem}.extracted"
        if marker.exists():
            continue
        print(f"Extracting archive: {archive}")
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                ensure_safe_extract_path(out_dir, member.name)
            tf.extractall(out_dir)
        marker.write_text("ok\n")

    for archive in sorted(raw_root.rglob("annotations.zip")):
        out_dir = archive.parent
        marker = out_dir / ".annotations.extracted"
        if marker.exists():
            continue
        print(f"Extracting archive: {archive}")
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                ensure_safe_extract_path(out_dir, member)
            zf.extractall(out_dir)
        marker.write_text("ok\n")


def discover_sequences(raw_root: Path, auto_extract_tars: bool = False) -> list[RawSequence]:
    if auto_extract_tars:
        extract_archives(raw_root)

    preextracted = discover_preextracted(raw_root)
    if preextracted:
        return preextracted

    sequences = discover_seq_vbb(raw_root)
    if sequences:
        return sequences

    maybe_print_archive_hint(raw_root)
    raise RuntimeError(
        f"No Caltech sequences found in {raw_root}. Expected either setXX/VYYY.seq with "
        "annotations/setXX/VYYY.vbb, or extracted setXX/VYYY/images + annotations dirs."
    )


def split_sequences(
    sequences: list[RawSequence],
    val_ratio: float,
    max_sequences: int,
) -> dict[str, list[RawSequence]]:
    sequences = sorted(sequences, key=lambda s: (s.set_name, s.video_name))
    if max_sequences > 0:
        sequences = sequences[:max_sequences]

    train_candidates = [s for s in sequences if s.set_name in TRAIN_SETS]
    test = [s for s in sequences if s.set_name in TEST_SETS]

    if len(train_candidates) >= 2 and val_ratio > 0:
        val_count = max(1, int(round(len(train_candidates) * val_ratio)))
        val = train_candidates[-val_count:]
        train = train_candidates[:-val_count]
    else:
        train = train_candidates
        val = []

    if not train:
        raise RuntimeError("No train sequences after split. Need at least one set00-set05 sequence.")
    if not val:
        print("Warning: val split is empty. Increase --max-sequences or lower --val-ratio.")

    return {"train": train, "val": val, "test": test}


def convert_preextracted_sequence(seq: RawSequence, args) -> tuple[dict, SequenceStats]:
    assert seq.images_dir is not None and seq.annotations_dir is not None
    image_paths = sorted_image_paths(seq.images_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {seq.images_dir}")

    frames = []
    times = []
    labels_obj = []
    labels_box = []
    stats = SequenceStats()
    fps = 30.0

    for ordinal, image_path in enumerate(tqdm(image_paths, desc=seq.name, leave=False)):
        frame_idx = image_index(image_path, ordinal)
        if frame_idx % args.frame_step != 0:
            continue

        frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        annotation_path = seq.annotations_dir / f"{image_path.stem}.json"
        boxes = read_json_boxes(annotation_path) if annotation_path.exists() else []
        obj, box = choose_single_target(
            boxes,
            width=width,
            height=height,
            mode=args.target_mode,
            min_box_height=args.min_box_height,
        )

        frames.append(resize_rgb(frame_rgb, args.image_size))
        times.append(frame_idx / fps)
        labels_obj.append(obj)
        labels_box.append(box)
        stats.add(obj, box)

    return build_npz_payload(seq, frames, times, labels_obj, labels_box, fps), stats


def convert_seq_vbb_sequence(seq: RawSequence, args) -> tuple[dict, SequenceStats]:
    assert seq.seq_path is not None and seq.vbb_path is not None
    header = read_seq_header(seq.seq_path)
    fps = float(header.get("fps", 30.0) or 30.0)
    boxes_by_frame = read_vbb_boxes(seq.vbb_path)

    frames = []
    times = []
    labels_obj = []
    labels_box = []
    stats = SequenceStats()

    for frame_idx, frame_rgb in tqdm(iter_seq_frames(seq.seq_path, args.frame_step), desc=seq.name, leave=False):
        height, width = frame_rgb.shape[:2]
        obj, box = choose_single_target(
            boxes_by_frame.get(frame_idx, []),
            width=width,
            height=height,
            mode=args.target_mode,
            min_box_height=args.min_box_height,
        )

        frames.append(resize_rgb(frame_rgb, args.image_size))
        times.append(frame_idx / fps)
        labels_obj.append(obj)
        labels_box.append(box)
        stats.add(obj, box)

    return build_npz_payload(seq, frames, times, labels_obj, labels_box, fps), stats


def build_npz_payload(
    seq: RawSequence,
    frames: list[np.ndarray],
    times: list[float],
    labels_obj: list[float],
    labels_box: list[np.ndarray],
    fps: float,
) -> dict:
    if not frames:
        raise RuntimeError(f"No frames converted for {seq.name}")
    return {
        "frames": np.array(frames, dtype=np.uint8),
        "times": np.array(times, dtype=np.float32),
        "labels_obj": np.array(labels_obj, dtype=np.float32),
        "labels_box": np.array(labels_box, dtype=np.float32),
        "source": np.array(seq.name),
        "fps": np.array(fps, dtype=np.float32),
    }


def write_sequence(seq: RawSequence, split: str, out_root: Path, payload: dict, stats: SequenceStats) -> dict:
    split_dir = out_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    out_path = split_dir / f"{seq.name}.npz"
    np.savez_compressed(out_path, **payload)
    return {
        "path": str(out_path.relative_to(out_root)),
        "sequence": seq.name,
        "split": split,
        "frames": stats.frames,
        "positive_frames": stats.positive_frames,
        "fps": float(payload["fps"]),
    }


def write_manifest(out_root: Path, split: str, rows: list[dict]) -> None:
    path = out_root / f"manifest_{split}.jsonl"
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def print_stats(split_stats: dict[str, SequenceStats], split_rows: dict[str, list[dict]]) -> None:
    total = SequenceStats()
    for split in ("train", "val", "test"):
        stats = split_stats[split]
        total.merge(stats)
        ratio = stats.positive_frames / stats.frames if stats.frames else 0.0
        avg_w = stats.box_width_sum / stats.positive_frames if stats.positive_frames else 0.0
        avg_h = stats.box_height_sum / stats.positive_frames if stats.positive_frames else 0.0
        print(
            f"{split}: sequences={len(split_rows[split])} frames={stats.frames} "
            f"positive_frames={stats.positive_frames} positive_ratio={ratio:.1%} "
            f"avg_box_size={avg_w:.3f}x{avg_h:.3f}"
        )

    ratio = total.positive_frames / total.frames if total.frames else 0.0
    avg_w = total.box_width_sum / total.positive_frames if total.positive_frames else 0.0
    avg_h = total.box_height_sum / total.positive_frames if total.positive_frames else 0.0
    print(
        f"total: sequences={sum(len(v) for v in split_rows.values())} frames={total.frames} "
        f"positive_frames={total.positive_frames} positive_ratio={ratio:.1%} "
        f"avg_box_size={avg_w:.3f}x{avg_h:.3f}"
    )


def convert(args) -> None:
    raw_root = Path(args.raw_root)
    out_root = Path(args.out)
    if not raw_root.exists():
        raise FileNotFoundError(f"Missing raw root: {raw_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    sequences = discover_sequences(raw_root, auto_extract_tars=args.extract_tars)
    splits = split_sequences(sequences, val_ratio=args.val_ratio, max_sequences=args.max_sequences)

    split_rows = {"train": [], "val": [], "test": []}
    split_stats = {"train": SequenceStats(), "val": SequenceStats(), "test": SequenceStats()}

    for split, split_sequences_ in splits.items():
        (out_root / split).mkdir(parents=True, exist_ok=True)
        for seq in split_sequences_:
            if seq.images_dir is not None:
                payload, stats = convert_preextracted_sequence(seq, args)
            else:
                payload, stats = convert_seq_vbb_sequence(seq, args)
            row = write_sequence(seq, split, out_root, payload, stats)
            split_rows[split].append(row)
            split_stats[split].merge(stats)

    for split, rows in split_rows.items():
        write_manifest(out_root, split, rows)

    print(f"Saved prepared dataset: {out_root}")
    print_stats(split_stats, split_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Caltech Pedestrian sequences for CNN+CfC training.")
    parser.add_argument("--raw-root", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--target-mode", choices=("largest", "union"), default="largest")
    parser.add_argument("--frame-step", type=int, default=3)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--min-box-height", type=float, default=10.0)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--extract-tars", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.frame_step < 1:
        raise ValueError("--frame-step must be >= 1")
    convert(args)


if __name__ == "__main__":
    main()

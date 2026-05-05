from __future__ import annotations

import json
import re
import struct
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

from cfc_video_demo.utils.video import IMG_EXTS


SET_RE = re.compile(r"set\d\d")
IMAGE_EXT_BY_FORMAT = {100: "raw", 102: "jpg", 201: "jpg", 1: "png", 2: "png"}


def is_person_label(label: str) -> bool:
    label = label.lower().strip()
    return label.startswith("person") or label == "people"


def find_set_name(path: Path) -> str | None:
    for part in path.parts:
        if SET_RE.fullmatch(part):
            return part
    return None


def sequence_name(set_name: str, video_name: str) -> str:
    return f"{set_name}_{video_name}"


def read_seq_header(path: str | Path) -> dict[str, float | int | str]:
    with Path(path).open("rb") as f:
        f.read(4)
        f.read(24)
        version = struct.unpack("<i", f.read(4))[0]
        header_length = struct.unpack("<i", f.read(4))[0]
        f.read(512)
        params = [struct.unpack("<i", f.read(4))[0] for _ in range(9)]
        fps = struct.unpack("<d", f.read(8))[0]
        f.read(432)

    fmt = params[5]
    return {
        "version": version,
        "header_length": header_length,
        "width": params[0],
        "height": params[1],
        "bit_depth": params[2],
        "format": fmt,
        "ext": IMAGE_EXT_BY_FORMAT.get(fmt, "jpg"),
        "num_frames": params[6],
        "fps": float(fps) if fps > 0 else 30.0,
    }


def iter_seq_frames(path: str | Path, frame_step: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    path = Path(path)
    header = read_seq_header(path)
    if header["ext"] not in {"jpg", "png"}:
        raise RuntimeError(f"Unsupported Caltech .seq image format={header['format']} in {path}")

    raw = path.read_bytes()
    pos = 1024
    extra = 8
    num_frames = int(header["num_frames"])

    for frame_idx in range(num_frames):
        if pos + 4 > len(raw):
            break
        size = struct.unpack_from("<I", raw, pos)[0]
        encoded = raw[pos + 4 : pos + size]
        pos += size + extra

        if frame_idx == 0 and pos < len(raw):
            marker = raw[pos]
            if marker != 0:
                pos -= 4
            else:
                extra += 8
                pos += 8

        if frame_idx % frame_step != 0:
            continue

        arr = np.frombuffer(encoded, dtype=np.uint8)
        frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise RuntimeError(f"Could not decode frame {frame_idx} from {path}")
        yield frame_idx, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def read_vbb_boxes(path: str | Path) -> dict[int, list[list[float]]]:
    vbb = loadmat(path)
    root = vbb["A"][0][0]
    obj_lists = root[1][0]
    obj_labels = [str(v[0]) for v in root[4][0]]

    boxes_by_frame: dict[int, list[list[float]]] = defaultdict(list)
    for frame_idx, objects in enumerate(obj_lists):
        if objects.shape[1] == 0:
            continue
        for obj_id, pos in zip(objects["id"][0], objects["pos"][0]):
            object_idx = int(obj_id[0][0]) - 1
            if object_idx < 0 or object_idx >= len(obj_labels):
                continue
            if not is_person_label(obj_labels[object_idx]):
                continue
            x, y, w, h = [float(v) for v in pos[0].tolist()]
            boxes_by_frame[frame_idx].append([x - 1.0, y - 1.0, w, h])
    return boxes_by_frame


def read_json_boxes(path: str | Path) -> list[list[float]]:
    items = json.loads(Path(path).read_text())
    boxes = []
    for item in items:
        if not is_person_label(str(item.get("lbl", ""))):
            continue
        pos = item.get("pos")
        if not pos or len(pos) < 4:
            continue
        boxes.append([float(pos[0]), float(pos[1]), float(pos[2]), float(pos[3])])
    return boxes


def image_index(path: Path, fallback: int) -> int:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else fallback


def sorted_image_paths(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


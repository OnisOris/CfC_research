from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resize_rgb(frame_rgb: np.ndarray, image_size: int) -> np.ndarray:
    return cv2.resize(frame_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)


def infer_fps(times: np.ndarray, fallback: float = 30.0) -> float:
    if len(times) < 2:
        return fallback
    dt = np.diff(times.astype(np.float32))
    dt = dt[dt > 1e-4]
    if len(dt) == 0:
        return fallback
    return float(1.0 / np.median(dt))


def write_mp4(frames_bgr: list[np.ndarray], out: str | Path, fps: float) -> None:
    if not frames_bgr:
        raise ValueError("No frames to write")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {out}")
    for frame in frames_bgr:
        writer.write(frame)
    writer.release()


def transcode_h264_if_available(path: str | Path) -> bool:
    path = Path(path)
    tmp = path.with_name(f".{path.stem}.h264-tmp{path.suffix}")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(tmp),
            ],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(path)
    return True


import argparse
import configparser
import csv
import pickle
import shutil
import subprocess
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import CfC
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


H, W = 96, 96


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device)
    except pickle.UnpicklingError:
        # Backward compatibility for older checkpoints that stored numpy arrays.
        # Only use this with checkpoints created locally by this demo.
        return torch.load(path, map_location=device, weights_only=False)


def resize_rgb(frame_bgr):
    frame = cv2.resize(frame_bgr, (W, H))
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def make_background(frames, bg_frames=60):
    n = min(bg_frames, len(frames))
    return np.median(frames[:n], axis=0).astype(np.uint8)


def diff_image(frame_rgb, bg_rgb):
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    bg_gray = cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2GRAY)

    diff = cv2.absdiff(gray, bg_gray)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    return diff


def label_from_diff(diff, threshold=35, min_area=40):
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.0, np.array([0, 0, 0, 0], dtype=np.float32), mask

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    if area < min_area:
        return 0.0, np.array([0, 0, 0, 0], dtype=np.float32), mask

    x, y, w, h = cv2.boundingRect(cnt)

    cx = (x + w / 2) / W
    cy = (y + h / 2) / H
    bw = w / W
    bh = h / H

    box = np.array([cx, cy, bw, bh], dtype=np.float32)
    return 1.0, box, mask


def load_labels(path, expected_len):
    data = np.load(path)
    labels_obj = data["labels_obj"].astype(np.float32)
    labels_box = data["labels_box"].astype(np.float32)

    if len(labels_obj) != expected_len or len(labels_box) != expected_len:
        raise ValueError(
            f"Label length mismatch: labels={len(labels_obj)}, frames={expected_len}"
        )

    return labels_obj, labels_box


def read_seqinfo(seq_dir):
    seqinfo = seq_dir / "seqinfo.ini"
    if not seqinfo.exists():
        return {}

    parser = configparser.ConfigParser()
    parser.read(seqinfo)
    if "Sequence" not in parser:
        return {}

    section = parser["Sequence"]
    return {
        "fps": section.getfloat("frameRate", fallback=30.0),
        "length": section.getint("seqLength", fallback=0),
        "ext": section.get("imExt", fallback=".jpg"),
    }


def mot_row_is_person(parts, include_static=False):
    if len(parts) < 7:
        return False

    active = float(parts[6]) != 0.0
    if not active:
        return False

    # MOT tracking ground truth often stores class and visibility in columns
    # 8 and 9. MOT17Det detection rows may only have 7 columns, so keep them.
    if len(parts) >= 9:
        class_id = int(float(parts[7]))
        person_classes = {1, 2}
        if include_static:
            person_classes.add(7)
        if class_id not in person_classes:
            return False

    return True


def load_mot_frame_boxes(gt_path, min_visibility=0.0, include_static=False):
    boxes_by_frame = {}
    with gt_path.open(newline="") as f:
        reader = csv.reader(f)
        for parts in reader:
            if not parts or not mot_row_is_person(parts, include_static=include_static):
                continue

            if len(parts) >= 9:
                visibility = float(parts[8])
                if visibility < min_visibility:
                    continue

            frame_id = int(float(parts[0]))
            left = float(parts[2])
            top = float(parts[3])
            width = float(parts[4])
            height = float(parts[5])
            if width <= 1 or height <= 1:
                continue

            boxes_by_frame.setdefault(frame_id, []).append((left, top, width, height))

    return boxes_by_frame


def mot_boxes_to_label(boxes, src_w, src_h, mode):
    if not boxes:
        return 0.0, np.array([0, 0, 0, 0], dtype=np.float32)

    if mode == "largest":
        left, top, width, height = max(boxes, key=lambda b: b[2] * b[3])
        x1, y1, x2, y2 = left, top, left + width, top + height
    elif mode == "union":
        x1 = min(box[0] for box in boxes)
        y1 = min(box[1] for box in boxes)
        x2 = max(box[0] + box[2] for box in boxes)
        y2 = max(box[1] + box[3] for box in boxes)
    else:
        raise ValueError(f"Unknown box mode: {mode}")

    x1 = np.clip(x1, 0, src_w)
    y1 = np.clip(y1, 0, src_h)
    x2 = np.clip(x2, 0, src_w)
    y2 = np.clip(y2, 0, src_h)

    if x2 - x1 < 2 or y2 - y1 < 2:
        return 0.0, np.array([0, 0, 0, 0], dtype=np.float32)

    box = np.array(
        [
            ((x1 + x2) / 2) / src_w,
            ((y1 + y2) / 2) / src_h,
            (x2 - x1) / src_w,
            (y2 - y1) / src_h,
        ],
        dtype=np.float32,
    )
    return 1.0, box


def first_present(row, names):
    for name in names:
        if name in row and row[name] != "":
            return row[name]
    return None


def load_frame_csv_boxes(path):
    boxes = []
    with Path(path).open(newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        try:
            has_header = csv.Sniffer().has_header(sample)
        except csv.Error:
            has_header = True

        if has_header:
            reader = csv.DictReader(f)
            for row in reader:
                x = first_present(row, ("x", "xmin", "left", "bb_left"))
                y = first_present(row, ("y", "ymin", "top", "bb_top"))
                w = first_present(row, ("w", "width", "bb_width"))
                h = first_present(row, ("h", "height", "bb_height"))
                xmax = first_present(row, ("xmax", "right"))
                ymax = first_present(row, ("ymax", "bottom"))

                if x is None or y is None:
                    boxes.append(None)
                    continue

                x = float(x)
                y = float(y)
                if w is not None and h is not None:
                    w = float(w)
                    h = float(h)
                elif xmax is not None and ymax is not None:
                    w = float(xmax) - x
                    h = float(ymax) - y
                else:
                    boxes.append(None)
                    continue

                boxes.append((x, y, w, h) if w > 1 and h > 1 else None)
        else:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 4:
                    boxes.append(None)
                    continue
                x, y, w, h = [float(v) for v in row[:4]]
                boxes.append((x, y, w, h) if w > 1 and h > 1 else None)

    return boxes


def csv_box_to_label(box, src_w, src_h):
    if box is None:
        return 0.0, np.array([0, 0, 0, 0], dtype=np.float32)
    return mot_boxes_to_label([box], src_w=src_w, src_h=src_h, mode="largest")


def resolve_video_path(path):
    path = Path(path)
    if path.exists():
        return path

    if path.suffix:
        return path

    for suffix in (".mp4", ".avi", ".mov", ".mkv"):
        candidate = path.with_suffix(suffix)
        if candidate.exists():
            return candidate

    return path


def box_to_rect(box):
    cx, cy, bw, bh = box
    x1 = int((cx - bw / 2) * W)
    y1 = int((cy - bh / 2) * H)
    x2 = int((cx + bw / 2) * W)
    y2 = int((cy + bh / 2) * H)
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))
    return x1, y1, x2, y2


def rect_to_box(x1, y1, x2, y2):
    x1, x2 = sorted((max(0, min(W - 1, x1)), max(0, min(W - 1, x2))))
    y1, y2 = sorted((max(0, min(H - 1, y1)), max(0, min(H - 1, y2))))

    if x2 - x1 < 2 or y2 - y1 < 2:
        return 0.0, np.array([0, 0, 0, 0], dtype=np.float32)

    box = np.array(
        [
            ((x1 + x2) / 2) / W,
            ((y1 + y2) / 2) / H,
            (x2 - x1) / W,
            (y2 - y1) / H,
        ],
        dtype=np.float32,
    )
    return 1.0, box


class WebcamSeqDataset(Dataset):
    def __init__(
        self,
        frames,
        times,
        bg,
        seq_len=8,
        diff_threshold=35,
        min_area=40,
        augment=False,
        aug_shift=16,
        aug_flip=True,
        labels_obj=None,
        labels_box=None,
    ):
        if len(frames) <= seq_len:
            raise ValueError(
                f"Need more than seq_len frames: got {len(frames)}, seq_len={seq_len}"
            )

        self.frames = frames
        self.times = times
        self.bg = bg
        self.seq_len = seq_len
        self.augment = augment
        self.aug_shift = aug_shift
        self.aug_flip = aug_flip

        self.diffs = []
        self.labels_obj = []
        self.labels_box = []

        use_manual_labels = labels_obj is not None and labels_box is not None

        for i, frame in enumerate(frames):
            diff = diff_image(frame, bg)
            if use_manual_labels:
                obj = labels_obj[i]
                box = labels_box[i]
            else:
                obj, box, _ = label_from_diff(
                    diff, threshold=diff_threshold, min_area=min_area
                )
            self.diffs.append(diff)
            self.labels_obj.append(obj)
            self.labels_box.append(box)

        self.diffs = np.array(self.diffs, dtype=np.uint8)
        self.labels_obj = np.array(self.labels_obj, dtype=np.float32)
        self.labels_box = np.array(self.labels_box, dtype=np.float32)

    def __len__(self):
        return len(self.frames) - self.seq_len

    def augment_window(self, x, obj, box):
        if self.aug_flip and np.random.random() < 0.5:
            x = x[:, :, ::-1].copy()
            if obj > 0:
                box = box.copy()
                box[0] = 1.0 - box[0]

        if self.aug_shift <= 0:
            return x, obj, box

        dx = np.random.randint(-self.aug_shift, self.aug_shift + 1)
        dy = np.random.randint(-self.aug_shift, self.aug_shift + 1)
        if dx == 0 and dy == 0:
            return x, obj, box

        shifted = np.empty_like(x)
        mat = np.float32([[1, 0, dx], [0, 1, dy]])
        for i, frame in enumerate(x):
            shifted[i] = cv2.warpAffine(
                frame,
                mat,
                (W, H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        if obj <= 0:
            return shifted, obj, box

        cx, cy, bw, bh = box
        x1 = (cx - bw / 2) * W + dx
        y1 = (cy - bh / 2) * H + dy
        x2 = (cx + bw / 2) * W + dx
        y2 = (cy + bh / 2) * H + dy

        x1 = np.clip(x1, 0, W)
        y1 = np.clip(y1, 0, H)
        x2 = np.clip(x2, 0, W)
        y2 = np.clip(y2, 0, H)

        if x2 - x1 < 2 or y2 - y1 < 2:
            return shifted, 0.0, np.array([0, 0, 0, 0], dtype=np.float32)

        new_box = np.array(
            [
                ((x1 + x2) / 2) / W,
                ((y1 + y2) / 2) / H,
                (x2 - x1) / W,
                (y2 - y1) / H,
            ],
            dtype=np.float32,
        )
        return shifted, obj, new_box

    def __getitem__(self, idx):
        start = idx
        end = idx + self.seq_len

        x = self.diffs[start:end].astype(np.float32) / 255.0

        t = self.times[start:end].astype(np.float32)
        dt = np.diff(t, prepend=t[0])
        if len(dt) > 1:
            dt[0] = np.median(dt[1:])
        dt = np.clip(dt, 1e-3, 0.2).astype(np.float32)

        obj = self.labels_obj[end - 1]
        box = self.labels_box[end - 1]

        if self.augment:
            x, obj, box = self.augment_window(x, obj, box)

        x = x[:, None, :, :]

        return (
            torch.tensor(x),
            torch.tensor(dt),
            torch.tensor(obj),
            torch.tensor(box),
        )


class CfcMotionDetector(nn.Module):
    def __init__(self, feat_dim=64, hidden=64):
        super().__init__()
        self.hidden = hidden

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, feat_dim),
            nn.ReLU(),
        )

        self.cfc = CfC(input_size=feat_dim, units=hidden)

        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
        )

    def forward(self, x, dt=None):
        b, t, c, h, w = x.shape

        x = x.reshape(b * t, c, h, w)
        x = self.add_coord_channels(x)
        z = self.encoder(x)
        z = z.reshape(b, t, -1)

        if dt is not None:
            if dt.ndim == 2:
                # ncps 1.0.1 squeezes the per-step timespan before passing it to
                # CfCCell. Expanding to hidden size keeps broadcasting valid for
                # batch sizes greater than one.
                dt = dt.unsqueeze(-1).expand(-1, -1, self.hidden)
            try:
                y, _hn = self.cfc(z, timespans=dt)
            except TypeError:
                try:
                    y, _hn = self.cfc(z, None, dt)
                except TypeError:
                    y, _hn = self.cfc(z)
        else:
            y, _hn = self.cfc(z)

        if y.ndim == 3:
            y = y[:, -1, :]

        out = self.head(y)

        obj_logit = out[:, 0]
        box = torch.sigmoid(out[:, 1:5])

        return obj_logit, box

    def add_coord_channels(self, x):
        b, _c, h, w = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(b, -1, -1, -1)
        return torch.cat([x, coords], dim=1)


def collect(args):
    cap = cv2.VideoCapture(args.cam)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index={args.cam}")

    frames = []
    times = []

    print("First 2-3 seconds: keep an empty background.")
    print("Then move an object in front of the camera. Press q to stop.")

    t0 = time.time()
    deadline = t0 + args.seconds

    while time.time() < deadline:
        ok, frame = cap.read()
        if not ok:
            continue

        rgb = resize_rgb(frame)
        frames.append(rgb)
        times.append(time.time() - t0)

        cv2.imshow("collect: press q to stop", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    frames = np.array(frames, dtype=np.uint8)
    times = np.array(times, dtype=np.float32)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, frames=frames, times=times)

    print(f"Saved: {args.out}")
    print(f"Frames: {len(frames)}")


def infer_fps(times, fallback=30.0):
    if len(times) < 2:
        return fallback
    dt = np.diff(times.astype(np.float32))
    dt = dt[dt > 1e-4]
    if len(dt) == 0:
        return fallback
    fps = 1.0 / float(np.median(dt))
    return max(1.0, min(120.0, fps))


def transcode_browser_mp4(src, dst):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "Browser-compatible MP4 export requires ffmpeg. "
            "Install ffmpeg or pass --no-browser-compatible to keep the raw OpenCV codec."
        )

    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-profile:v",
        "high",
        "-level",
        "4.0",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(dst),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed while creating browser-compatible MP4:\n{proc.stderr}")


def export_video(args):
    data = np.load(args.data)
    frames = data["frames"]
    times = data["times"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fps = args.fps if args.fps > 0 else infer_fps(times)
    size = (args.width, args.height)
    use_transcode = args.browser_compatible and out.suffix.lower() == ".mp4"
    raw_out = out.with_name(f".{out.stem}.opencv-tmp{out.suffix}") if use_transcode else out

    fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
    writer = cv2.VideoWriter(str(raw_out), fourcc, fps, size)

    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {raw_out}")

    for frame_rgb in frames:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if size != (W, H):
            frame_bgr = cv2.resize(frame_bgr, size, interpolation=cv2.INTER_NEAREST)
        writer.write(frame_bgr)

    writer.release()
    if use_transcode:
        transcode_browser_mp4(raw_out, out)
        raw_out.unlink(missing_ok=True)

    print(f"Saved video: {out}")
    if use_transcode:
        print("Codec: H.264/AVC (browser-compatible MP4)")
    print(f"Frames: {len(frames)}")
    print(f"FPS: {fps:.2f}")
    print(f"Size: {size[0]}x{size[1]}")


def export_frames(args):
    data = np.load(args.data)
    frames = data["frames"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    size = (args.width, args.height)
    saved = 0
    for i, frame_rgb in enumerate(frames):
        if i % args.every != 0:
            continue
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if size != (W, H):
            frame_bgr = cv2.resize(frame_bgr, size, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out_dir / f"frame_{i:05d}.jpg"), frame_bgr)
        saved += 1

    print(f"Saved frames: {out_dir}")
    print(f"Frames: {saved}/{len(frames)}")


def prepare_mot(args):
    seq_dir = Path(args.seq)
    img_dir = seq_dir / "img1"
    gt_path = seq_dir / "gt" / "gt.txt"

    if not img_dir.exists():
        raise FileNotFoundError(f"Missing MOT image directory: {img_dir}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing MOT ground-truth file: {gt_path}")

    seqinfo = read_seqinfo(seq_dir)
    fps = args.fps or seqinfo.get("fps", 30.0)
    ext = args.ext or seqinfo.get("ext", ".jpg")

    frame_paths = sorted(img_dir.glob(f"*{ext}"))
    if args.max_frames:
        frame_paths = frame_paths[: args.max_frames]
    if not frame_paths:
        raise RuntimeError(f"No frames matching *{ext} in {img_dir}")

    boxes_by_frame = load_mot_frame_boxes(
        gt_path,
        min_visibility=args.min_visibility,
        include_static=args.include_static,
    )

    frames = []
    times = []
    labels_obj = []
    labels_box = []

    src_w = None
    src_h = None
    for i, frame_path in enumerate(tqdm(frame_paths, desc="prepare MOT")):
        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            raise RuntimeError(f"Could not read frame: {frame_path}")

        if src_w is None or src_h is None:
            src_h, src_w = frame_bgr.shape[:2]

        frame_id = int(frame_path.stem)
        obj, box = mot_boxes_to_label(
            boxes_by_frame.get(frame_id, []),
            src_w=src_w,
            src_h=src_h,
            mode=args.box_mode,
        )

        frames.append(resize_rgb(frame_bgr))
        times.append(i / fps)
        labels_obj.append(obj)
        labels_box.append(box)

    out_data = Path(args.out_data)
    out_labels = Path(args.out_labels)
    out_data.parent.mkdir(parents=True, exist_ok=True)
    out_labels.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_data,
        frames=np.array(frames, dtype=np.uint8),
        times=np.array(times, dtype=np.float32),
        source_mot_sequence=str(seq_dir),
        fps=np.array(fps, dtype=np.float32),
    )
    np.savez_compressed(
        out_labels,
        labels_obj=np.array(labels_obj, dtype=np.float32),
        labels_box=np.array(labels_box, dtype=np.float32),
        source_data=str(out_data),
        source_mot_sequence=str(seq_dir),
        box_mode=args.box_mode,
    )

    positives = int(np.sum(labels_obj))
    print(f"Saved data: {out_data}")
    print(f"Saved labels: {out_labels}")
    print(f"Frames: {len(frames)}")
    print(f"Positive frames: {positives}/{len(frames)} ({positives / len(frames):.1%})")
    print(f"Box mode: {args.box_mode}")


def prepare_video_csv(args):
    video_path = resolve_video_path(args.video)
    csv_path = Path(args.csv)
    if not video_path.exists():
        raise FileNotFoundError(f"Missing video: {video_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV labels: {csv_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = args.fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    labels = load_frame_csv_boxes(csv_path)

    frames = []
    times = []
    labels_obj = []
    labels_box = []

    frame_idx = 0
    saved_idx = 0
    pbar = tqdm(desc="prepare video CSV")
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if args.max_frames and frame_idx >= args.max_frames:
            break

        if frame_idx % args.every == 0:
            src_h, src_w = frame_bgr.shape[:2]
            obj, box = csv_box_to_label(
                labels[frame_idx] if frame_idx < len(labels) else None,
                src_w=src_w,
                src_h=src_h,
            )

            frames.append(resize_rgb(frame_bgr))
            times.append(frame_idx / fps)
            labels_obj.append(obj)
            labels_box.append(box)
            saved_idx += 1

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from video: {video_path}")

    out_data = Path(args.out_data)
    out_labels = Path(args.out_labels)
    out_data.parent.mkdir(parents=True, exist_ok=True)
    out_labels.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_data,
        frames=np.array(frames, dtype=np.uint8),
        times=np.array(times, dtype=np.float32),
        source_video=str(video_path),
        source_csv=str(csv_path),
        fps=np.array(fps, dtype=np.float32),
    )
    np.savez_compressed(
        out_labels,
        labels_obj=np.array(labels_obj, dtype=np.float32),
        labels_box=np.array(labels_box, dtype=np.float32),
        source_data=str(out_data),
        source_video=str(video_path),
        source_csv=str(csv_path),
    )

    positives = int(np.sum(labels_obj))
    print(f"Saved data: {out_data}")
    print(f"Saved labels: {out_labels}")
    print(f"Read video frames: {frame_idx}")
    print(f"Saved frames: {saved_idx}")
    print(f"Positive frames: {positives}/{len(frames)} ({positives / len(frames):.1%})")


def print_label_stats(ds):
    obj = ds.labels_obj
    boxes = ds.labels_box[obj > 0]

    print(f"Label positive frames: {int(obj.sum())}/{len(obj)} ({obj.mean():.1%})")
    if len(boxes) == 0:
        print("No positive boxes found. Lower --diff-th or collect clearer motion.")
        return

    print(
        "Box mean cx cy w h: "
        + " ".join(f"{x:.3f}" for x in boxes.mean(axis=0))
    )
    print(
        "Box std  cx cy w h: "
        + " ".join(f"{x:.3f}" for x in boxes.std(axis=0))
    )
    print(
        "Box min  cx cy w h: "
        + " ".join(f"{x:.3f}" for x in boxes.min(axis=0))
    )
    print(
        "Box max  cx cy w h: "
        + " ".join(f"{x:.3f}" for x in boxes.max(axis=0))
    )


def train(args):
    data = np.load(args.data)
    frames = data["frames"]
    times = data["times"]

    bg = make_background(frames, bg_frames=args.bg_frames)
    labels_obj = None
    labels_box = None
    if args.labels:
        labels_obj, labels_box = load_labels(args.labels, len(frames))

    ds = WebcamSeqDataset(
        frames,
        times,
        bg,
        seq_len=args.seq_len,
        diff_threshold=args.diff_th,
        min_area=args.min_area,
        augment=not args.no_augment,
        aug_shift=args.aug_shift,
        aug_flip=not args.no_aug_flip,
        labels_obj=labels_obj,
        labels_box=labels_box,
    )
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    model = CfcMotionDetector(feat_dim=args.feat_dim, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Device: {device}")
    print(f"Samples: {len(ds)}")
    if args.labels:
        print(f"Labels: {args.labels}")
    print_label_stats(ds)

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_bce = 0.0
        total_box = 0.0

        pbar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")

        for x, dt, obj, box in pbar:
            x = x.to(device)
            dt = dt.to(device)
            obj = obj.to(device)
            box = box.to(device)

            obj_logit, pred_box = model(x, dt)

            loss_obj = F.binary_cross_entropy_with_logits(obj_logit, obj)

            per_sample_box = F.smooth_l1_loss(
                pred_box, box, reduction="none"
            ).mean(dim=1)
            loss_box = (per_sample_box * obj).sum() / (obj.sum() + 1e-6)

            loss = loss_obj + args.box_weight * loss_box

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            total_bce += loss_obj.item()
            total_box += loss_box.item()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                obj=f"{loss_obj.item():.4f}",
                box=f"{loss_box.item():.4f}",
            )

        print(
            f"epoch={epoch} "
            f"loss={total_loss / len(loader):.4f} "
            f"obj={total_bce / len(loader):.4f} "
            f"box={total_box / len(loader):.4f}"
        )

    ckpt = {
        "model": model.state_dict(),
        "background": torch.from_numpy(bg),
        "seq_len": args.seq_len,
        "feat_dim": args.feat_dim,
        "hidden": args.hidden,
        "arch": "coord-spatial-cnn-cfc-v3",
        "diff_threshold": args.diff_th,
        "min_area": args.min_area,
        "H": H,
        "W": W,
    }

    torch.save(ckpt, args.model)
    print(f"Saved model: {args.model}")


@torch.no_grad()
def demo(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    ckpt = load_checkpoint(args.model, device)

    model = CfcMotionDetector(
        feat_dim=ckpt["feat_dim"],
        hidden=ckpt["hidden"],
    ).to(device)

    try:
        model.load_state_dict(ckpt["model"])
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint architecture does not match current code. "
            "Retrain the model with `uv run cfc-motion-detector train ...`."
        ) from exc
    model.eval()

    seq_len = ckpt["seq_len"]

    cap = cv2.VideoCapture(args.cam)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index={args.cam}")

    if args.live_bg_frames > 0:
        print(
            f"Calibrating live background from {args.live_bg_frames} frames. "
            "Keep the target area empty/still."
        )
        bg_frames = []
        while len(bg_frames) < args.live_bg_frames:
            ok, frame_bgr = cap.read()
            if not ok:
                continue

            bg_frames.append(resize_rgb(frame_bgr))
            frame_show = cv2.resize(frame_bgr, (W * 4, H * 4))
            cv2.putText(
                frame_show,
                f"Calibrating background {len(bg_frames)}/{args.live_bg_frames}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )
            cv2.imshow("CfC moving object detector", frame_show)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return

        bg = make_background(np.array(bg_frames, dtype=np.uint8), bg_frames=len(bg_frames))
    else:
        bg = ckpt["background"]
        if isinstance(bg, torch.Tensor):
            bg = bg.cpu().numpy()

    buf = deque(maxlen=seq_len)
    time_buf = deque(maxlen=seq_len)

    print("Move an object in front of the camera. Press q to exit.")

    t0 = time.time()

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            continue

        rgb = resize_rgb(frame_bgr)
        diff = diff_image(rgb, bg)

        now = time.time() - t0

        buf.append(diff)
        time_buf.append(now)

        frame_show = cv2.resize(frame_bgr, (W * 4, H * 4))

        if len(buf) == seq_len:
            x = np.array(buf, dtype=np.float32) / 255.0
            x = x[:, None, :, :]
            x = torch.tensor(x).unsqueeze(0).to(device)

            t = np.array(time_buf, dtype=np.float32)
            dt = np.diff(t, prepend=t[0])
            if len(dt) > 1:
                dt[0] = np.median(dt[1:])
            dt = np.clip(dt, 1e-3, 0.2).astype(np.float32)
            dt = torch.tensor(dt).unsqueeze(0).to(device)

            obj_logit, box = model(x, dt)
            prob = torch.sigmoid(obj_logit)[0].item()
            cx, cy, bw, bh = box[0].cpu().numpy()

            if prob > args.th:
                x1 = int((cx - bw / 2) * W)
                y1 = int((cy - bh / 2) * H)
                x2 = int((cx + bw / 2) * W)
                y2 = int((cy + bh / 2) * H)

                x1 = max(0, min(W - 1, x1))
                y1 = max(0, min(H - 1, y1))
                x2 = max(0, min(W - 1, x2))
                y2 = max(0, min(H - 1, y2))

                scale = 4
                cv2.rectangle(
                    frame_show,
                    (x1 * scale, y1 * scale),
                    (x2 * scale, y2 * scale),
                    (0, 0, 255),
                    2,
                )

            if args.show_teacher:
                teacher_obj, teacher_box, _mask = label_from_diff(
                    diff, threshold=args.diff_th, min_area=args.min_area
                )
                if teacher_obj > 0:
                    tcx, tcy, tbw, tbh = teacher_box
                    tx1 = int((tcx - tbw / 2) * W)
                    ty1 = int((tcy - tbh / 2) * H)
                    tx2 = int((tcx + tbw / 2) * W)
                    ty2 = int((tcy + tbh / 2) * H)
                    scale = 4
                    cv2.rectangle(
                        frame_show,
                        (tx1 * scale, ty1 * scale),
                        (tx2 * scale, ty2 * scale),
                        (0, 255, 0),
                        1,
                    )

            cv2.putText(
                frame_show,
                f"CfC motion prob: {prob:.2f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv2.imshow("CfC moving object detector", frame_show)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def inspect(args):
    data = np.load(args.data)
    frames = data["frames"]
    times = data["times"]

    bg = make_background(frames, bg_frames=args.bg_frames)
    labels_obj = None
    labels_box = None
    if args.labels:
        labels_obj, labels_box = load_labels(args.labels, len(frames))

    ds = WebcamSeqDataset(
        frames,
        times,
        bg,
        seq_len=args.seq_len,
        diff_threshold=args.diff_th,
        min_area=args.min_area,
        labels_obj=labels_obj,
        labels_box=labels_box,
    )
    print_label_stats(ds)
    if args.labels:
        print(f"Labels: {args.labels}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    scale = args.scale
    for i, frame_rgb in enumerate(frames):
        if i % args.every != 0:
            continue

        diff = diff_image(frame_rgb, bg)
        auto_obj, _auto_box, mask = label_from_diff(
            diff, threshold=args.diff_th, min_area=args.min_area
        )
        if args.labels:
            obj = labels_obj[i]
            box = labels_box[i]
        else:
            obj = auto_obj
            box = _auto_box

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_show = cv2.resize(frame_bgr, (W * scale, H * scale))

        if obj > 0:
            cx, cy, bw, bh = box
            x1 = int((cx - bw / 2) * W)
            y1 = int((cy - bh / 2) * H)
            x2 = int((cx + bw / 2) * W)
            y2 = int((cy + bh / 2) * H)
            cv2.rectangle(
                frame_show,
                (x1 * scale, y1 * scale),
                (x2 * scale, y2 * scale),
                (0, 0, 255),
                2,
            )

        diff_show = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        diff_show = cv2.resize(diff_show, (W * scale, H * scale))
        mask_show = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_show = cv2.resize(mask_show, (W * scale, H * scale))

        panel = np.concatenate([frame_show, diff_show, mask_show], axis=1)
        cv2.imwrite(str(out_dir / f"label_{i:05d}.jpg"), panel)

    print(f"Saved label previews to: {out_dir}")


def annotate(args):
    data = np.load(args.data)
    frames = data["frames"]
    times = data["times"]

    bg = make_background(frames, bg_frames=args.bg_frames)
    labels_obj = np.zeros(len(frames), dtype=np.float32)
    labels_box = np.zeros((len(frames), 4), dtype=np.float32)

    if args.resume and Path(args.resume).exists():
        labels_obj, labels_box = load_labels(args.resume, len(frames))
        print(f"Loaded labels: {args.resume}")
    elif args.init_auto:
        for i, frame in enumerate(frames):
            diff = diff_image(frame, bg)
            labels_obj[i], labels_box[i], _ = label_from_diff(
                diff, threshold=args.diff_th, min_area=args.min_area
            )
        print("Initialized labels from background-diff teacher.")

    state = {
        "idx": 0,
        "drawing": False,
        "start": None,
        "current": None,
        "dirty": False,
    }
    scale = args.scale

    def save_labels():
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = out.with_name(f"{out.stem}.bak-{timestamp}{out.suffix}")
            shutil.copy2(out, backup)
            print(f"Backup labels: {backup}")
        np.savez_compressed(
            out,
            labels_obj=labels_obj,
            labels_box=labels_box,
            source_data=str(args.data),
            times=times,
        )
        state["dirty"] = False
        print(f"Saved labels: {out}")

    def mouse(event, x, y, _flags, _param):
        px = max(0, min(W - 1, x // scale))
        py = max(0, min(H - 1, y // scale))

        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["start"] = (px, py)
            state["current"] = (px, py)
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["current"] = (px, py)
        elif event == cv2.EVENT_LBUTTONUP and state["drawing"]:
            state["drawing"] = False
            x1, y1 = state["start"]
            obj, box = rect_to_box(x1, y1, px, py)
            labels_obj[state["idx"]] = obj
            labels_box[state["idx"]] = box
            state["current"] = None
            state["dirty"] = True

    cv2.namedWindow("manual bbox annotation")
    cv2.setMouseCallback("manual bbox annotation", mouse)

    print("Manual annotation controls:")
    print("  mouse drag: set bbox")
    print("  n/space: next frame")
    print("  p: previous frame")
    print("  f: forward 10 frames")
    print("  b: back 10 frames")
    print("  d: mark no object")
    print("  a: replace current frame with auto-label")
    print("  s: save")
    print("  w: save and quit")
    print("  q: quit without saving")

    while True:
        idx = state["idx"]
        frame_bgr = cv2.cvtColor(frames[idx], cv2.COLOR_RGB2BGR)
        frame_show = cv2.resize(frame_bgr, (W * scale, H * scale))

        if labels_obj[idx] > 0:
            x1, y1, x2, y2 = box_to_rect(labels_box[idx])
            cv2.rectangle(
                frame_show,
                (x1 * scale, y1 * scale),
                (x2 * scale, y2 * scale),
                (0, 0, 255),
                2,
            )

        if state["drawing"] and state["start"] and state["current"]:
            x1, y1 = state["start"]
            x2, y2 = state["current"]
            cv2.rectangle(
                frame_show,
                (x1 * scale, y1 * scale),
                (x2 * scale, y2 * scale),
                (0, 255, 0),
                1,
            )

        status = "obj" if labels_obj[idx] > 0 else "empty"
        dirty = "*" if state["dirty"] else ""
        text = f"{idx + 1}/{len(frames)} {status}{dirty} | n/p/f/b d a s w q"
        cv2.putText(
            frame_show,
            text,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
        )
        cv2.imshow("manual bbox annotation", frame_show)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord("n"), ord(" "), 83):
            state["idx"] = min(len(frames) - 1, idx + 1)
        elif key in (ord("p"), 81):
            state["idx"] = max(0, idx - 1)
        elif key == ord("f"):
            state["idx"] = min(len(frames) - 1, idx + 10)
        elif key == ord("b"):
            state["idx"] = max(0, idx - 10)
        elif key == ord("d"):
            labels_obj[idx] = 0.0
            labels_box[idx] = np.array([0, 0, 0, 0], dtype=np.float32)
            state["dirty"] = True
        elif key == ord("a"):
            diff = diff_image(frames[idx], bg)
            labels_obj[idx], labels_box[idx], _ = label_from_diff(
                diff, threshold=args.diff_th, min_area=args.min_area
            )
            state["dirty"] = True
        elif key == ord("s"):
            save_labels()
        elif key == ord("w"):
            save_labels()
            break
        elif key == ord("q"):
            if state["dirty"]:
                print("Quit without saving unsaved annotation changes.")
            break

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("collect")
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--seconds", type=int, default=45)
    p.add_argument("--out", type=str, default="data/webcam_motion.npz")
    p.set_defaults(func=collect)

    p = sub.add_parser("export-video")
    p.add_argument("--data", type=str, default="data/webcam_motion.npz")
    p.add_argument("--out", type=str, default="data/webcam_motion.mp4")
    p.add_argument("--fps", type=float, default=0.0)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--fourcc", type=str, default="mp4v")
    p.add_argument("--browser-compatible", action=argparse.BooleanOptionalAction, default=True)
    p.set_defaults(func=export_video)

    p = sub.add_parser("export-frames")
    p.add_argument("--data", type=str, default="data/webcam_motion.npz")
    p.add_argument("--out", type=str, default="data/frames")
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--height", type=int, default=384)
    p.set_defaults(func=export_frames)

    p = sub.add_parser("prepare-mot")
    p.add_argument("--seq", type=str, required=True)
    p.add_argument("--out-data", type=str, default="data/mot17_person_seq.npz")
    p.add_argument("--out-labels", type=str, default="data/mot17_person_labels.npz")
    p.add_argument("--fps", type=float, default=0.0)
    p.add_argument("--ext", type=str, default=None)
    p.add_argument("--box-mode", choices=("largest", "union"), default="largest")
    p.add_argument("--min-visibility", type=float, default=0.2)
    p.add_argument("--include-static", action="store_true")
    p.add_argument("--max-frames", type=int, default=0)
    p.set_defaults(func=prepare_mot)

    p = sub.add_parser("prepare-video-csv")
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--out-data", type=str, default="data/pedestrian_video_seq.npz")
    p.add_argument("--out-labels", type=str, default="data/pedestrian_video_labels.npz")
    p.add_argument("--fps", type=float, default=0.0)
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0)
    p.set_defaults(func=prepare_video_csv)

    p = sub.add_parser("train")
    p.add_argument("--data", type=str, default="data/webcam_motion.npz")
    p.add_argument("--labels", type=str, default=None)
    p.add_argument("--model", type=str, default="cfc_motion_detector.pt")
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--bg-frames", type=int, default=60)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--feat-dim", type=int, default=64)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--diff-th", type=int, default=35)
    p.add_argument("--min-area", type=int, default=40)
    p.add_argument("--box-weight", type=float, default=10.0)
    p.add_argument("--aug-shift", type=int, default=18)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-aug-flip", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=train)

    p = sub.add_parser("inspect")
    p.add_argument("--data", type=str, default="data/webcam_motion.npz")
    p.add_argument("--labels", type=str, default=None)
    p.add_argument("--out", type=str, default="data/label_preview")
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--bg-frames", type=int, default=60)
    p.add_argument("--diff-th", type=int, default=35)
    p.add_argument("--min-area", type=int, default=40)
    p.add_argument("--every", type=int, default=5)
    p.add_argument("--scale", type=int, default=4)
    p.set_defaults(func=inspect)

    p = sub.add_parser("annotate")
    p.add_argument("--data", type=str, default="data/webcam_motion.npz")
    p.add_argument("--out", type=str, default="data/manual_labels.npz")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--bg-frames", type=int, default=60)
    p.add_argument("--diff-th", type=int, default=35)
    p.add_argument("--min-area", type=int, default=40)
    p.add_argument("--scale", type=int, default=6)
    p.add_argument("--init-auto", action="store_true")
    p.set_defaults(func=annotate)

    p = sub.add_parser("demo")
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--model", type=str, default="cfc_motion_detector.pt")
    p.add_argument("--th", type=float, default=0.5)
    p.add_argument("--show-teacher", action="store_true")
    p.add_argument("--diff-th", type=int, default=35)
    p.add_argument("--min-area", type=int, default=40)
    p.add_argument("--live-bg-frames", type=int, default=60)
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=demo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

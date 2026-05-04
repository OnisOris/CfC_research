import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import CfC
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}


def load_checkpoint(path, device):
    return torch.load(path, map_location=device)


def read_simple_yolo_yaml(path):
    data = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip("'\"")
    root = Path(data.get("path", Path(path).parent)).expanduser()
    if not root.is_absolute():
        root = (Path(path).parent / root).resolve()
    return {
        "root": root,
        "train": data.get("train", "train/images"),
        "val": data.get("val", data.get("valid", "valid/images")),
        "test": data.get("test", "test/images"),
    }


def split_image_dir(dataset_cfg, split):
    key = "val" if split == "valid" else split
    return dataset_cfg["root"] / dataset_cfg[key]


def image_paths(image_dir, max_images=0):
    paths = sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in IMG_EXTS)
    if max_images:
        paths = paths[:max_images]
    if not paths:
        raise RuntimeError(f"No images found in {image_dir}")
    return paths


def find_first(root, suffixes):
    matches = sorted(
        p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in suffixes
    )
    if not matches:
        raise RuntimeError(f"No files with suffixes {sorted(suffixes)} found in {root}")
    return matches[0]


def yolo_label_path(image_path):
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def load_yolo_boxes(image_path):
    label_path = yolo_label_path(image_path)
    boxes = []
    if not label_path.exists():
        return boxes
    for raw in label_path.read_text().splitlines():
        parts = raw.split()
        if len(parts) < 5:
            continue
        cls, cx, cy, bw, bh = parts[:5]
        if int(float(cls)) != 0:
            continue
        box = np.array([float(cx), float(cy), float(bw), float(bh)], dtype=np.float32)
        if box[2] > 0 and box[3] > 0:
            boxes.append(box)
    return boxes


def parse_okutama_labels(path, include_generated=True):
    boxes_by_frame = {}
    with Path(path).open() as f:
        for raw in f:
            parts = raw.strip().split()
            if len(parts) < 10:
                continue

            x1 = float(parts[1])
            y1 = float(parts[2])
            x2 = float(parts[3])
            y2 = float(parts[4])
            frame_id = int(float(parts[5]))
            lost = int(float(parts[6]))
            generated = int(float(parts[8]))
            label = parts[9].strip('"').lower()

            if lost or label != "person":
                continue
            if generated and not include_generated:
                continue
            if x2 <= x1 or y2 <= y1:
                continue

            boxes_by_frame.setdefault(frame_id, []).append((x1, y1, x2, y2))
    return boxes_by_frame


def write_yolo_label(path, boxes, width, height):
    lines = []
    for x1, y1, x2, y2 in boxes:
        x1 = np.clip(x1, 0, width)
        x2 = np.clip(x2, 0, width)
        y1 = np.clip(y1, 0, height)
        y2 = np.clip(y2, 0, height)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        cx = ((x1 + x2) / 2) / width
        cy = ((y1 + y2) / 2) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        lines.append(f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""))


def boxes_to_target(boxes, mode):
    if not boxes:
        return 0.0, np.zeros(4, dtype=np.float32)

    boxes = np.array(boxes, dtype=np.float32)
    if mode == "largest":
        areas = boxes[:, 2] * boxes[:, 3]
        box = boxes[int(np.argmax(areas))]
        return 1.0, box.astype(np.float32)
    if mode == "union":
        x1 = np.min(boxes[:, 0] - boxes[:, 2] / 2)
        y1 = np.min(boxes[:, 1] - boxes[:, 3] / 2)
        x2 = np.max(boxes[:, 0] + boxes[:, 2] / 2)
        y2 = np.max(boxes[:, 1] + boxes[:, 3] / 2)
        x1, y1, x2, y2 = np.clip([x1, y1, x2, y2], 0.0, 1.0)
        if x2 <= x1 or y2 <= y1:
            return 0.0, np.zeros(4, dtype=np.float32)
        return 1.0, np.array(
            [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1],
            dtype=np.float32,
        )
    raise ValueError(f"Unknown box mode: {mode}")


def prepare_okutama(args):
    root = Path(args.source)
    video_path = Path(args.video) if args.video else find_first(root, VIDEO_EXTS)
    label_path = Path(args.labels) if args.labels else find_first(root, {".txt"})
    out_root = Path(args.out)

    boxes_by_frame = parse_okutama_labels(
        label_path, include_generated=not args.no_generated
    )
    if not boxes_by_frame:
        raise RuntimeError(f"No usable person labels found in {label_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    train_img = out_root / "train" / "images"
    train_lbl = out_root / "train" / "labels"
    val_img = out_root / "valid" / "images"
    val_lbl = out_root / "valid" / "labels"
    for path in (train_img, train_lbl, val_img, val_lbl):
        path.mkdir(parents=True, exist_ok=True)

    max_frames = args.max_frames if args.max_frames > 0 else total
    train_cut = int(max_frames * args.train_ratio)
    written = positives = 0
    frame_idx = 0

    pbar = tqdm(total=max_frames if max_frames else None, desc="prepare Okutama")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames and frame_idx >= max_frames:
            break

        if frame_idx % args.every == 0:
            split_img = train_img if frame_idx < train_cut else val_img
            split_lbl = train_lbl if frame_idx < train_cut else val_lbl
            stem = f"frame_{frame_idx:06d}"
            image_path = split_img / f"{stem}.jpg"
            label_out = split_lbl / f"{stem}.txt"

            cv2.imwrite(str(image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            boxes = boxes_by_frame.get(frame_idx, [])
            if boxes:
                positives += 1
            write_yolo_label(label_out, boxes, width, height)
            written += 1

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    yaml_path = out_root / "okutama.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out_root.resolve()}",
                "train: train/images",
                "val: valid/images",
                "test: valid/images",
                "",
                "names:",
                "  0: human",
                "",
            ]
        )
    )

    print(f"Video: {video_path}")
    print(f"Labels: {label_path}")
    print(f"FPS: {fps:.2f}")
    print(f"Source size: {width}x{height}")
    print(f"Saved frames: {written}")
    print(f"Positive frames: {positives}/{written}")
    print(f"Saved dataset: {out_root}")
    print(f"Saved yaml: {yaml_path}")


class AerialYoloSeqDataset(Dataset):
    def __init__(
        self,
        paths,
        image_size=640,
        seq_len=1,
        box_mode="union",
        fps=5.0,
        augment=False,
    ):
        if len(paths) < seq_len:
            raise ValueError(f"Need at least seq_len images: {len(paths)} < {seq_len}")
        self.paths = paths
        self.image_size = image_size
        self.seq_len = seq_len
        self.box_mode = box_mode
        self.fps = fps
        self.augment = augment

    def __len__(self):
        return len(self.paths) - self.seq_len + 1

    def load_image(self, path):
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Could not read image: {path}")
        frame = cv2.resize(frame, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def __getitem__(self, idx):
        seq_paths = self.paths[idx : idx + self.seq_len]
        frames = np.array([self.load_image(path) for path in seq_paths], dtype=np.float32) / 255.0
        obj, box = boxes_to_target(load_yolo_boxes(seq_paths[-1]), self.box_mode)

        if self.augment and np.random.random() < 0.5:
            frames = frames[:, :, ::-1].copy()
            if obj > 0:
                box = box.copy()
                box[0] = 1.0 - box[0]

        x = frames.transpose(0, 3, 1, 2)
        dt = np.full(self.seq_len, 1.0 / self.fps, dtype=np.float32)
        return (
            torch.tensor(x),
            torch.tensor(dt),
            torch.tensor(obj, dtype=torch.float32),
            torch.tensor(box, dtype=torch.float32),
        )


class CfcAerialDetector(nn.Module):
    def __init__(self, image_size=640, feat_dim=128, hidden=128):
        super().__init__()
        self.hidden = hidden
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 24, 5, stride=2, padding=2),
            nn.BatchNorm2d(24),
            nn.SiLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(),
            nn.Conv2d(96, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, feat_dim),
            nn.SiLU(),
        )
        self.cfc = CfC(input_size=feat_dim, units=hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.SiLU(),
            nn.Linear(128, 5),
        )

    def add_coord_channels(self, x):
        b, _c, h, w = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(b, -1, -1, -1)
        return torch.cat([x, coords], dim=1)

    def forward(self, x, dt=None):
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
        return out[:, 0], torch.sigmoid(out[:, 1:5])


def train_one_epoch(model, loader, opt, device, box_weight):
    model.train()
    totals = {"loss": 0.0, "obj": 0.0, "box": 0.0}
    pbar = tqdm(loader, desc="train")
    for x, dt, obj, box in pbar:
        x, dt, obj, box = x.to(device), dt.to(device), obj.to(device), box.to(device)
        obj_logit, pred_box = model(x, dt)
        loss_obj = F.binary_cross_entropy_with_logits(obj_logit, obj)
        per_box = F.smooth_l1_loss(pred_box, box, reduction="none").mean(dim=1)
        loss_box = (per_box * obj).sum() / (obj.sum() + 1e-6)
        loss = loss_obj + box_weight * loss_box

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        totals["loss"] += loss.item()
        totals["obj"] += loss_obj.item()
        totals["box"] += loss_box.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", obj=f"{loss_obj.item():.4f}", box=f"{loss_box.item():.4f}")
    n = max(1, len(loader))
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def evaluate(model, loader, device, th=0.5):
    model.eval()
    tp = fp = fn = 0
    for x, dt, obj, _box in tqdm(loader, desc="val"):
        x, dt, obj = x.to(device), dt.to(device), obj.to(device)
        obj_logit, _pred_box = model(x, dt)
        pred = torch.sigmoid(obj_logit) > th
        target = obj > 0.5
        tp += int((pred & target).sum().item())
        fp += int((pred & ~target).sum().item())
        fn += int((~pred & target).sum().item())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return precision, recall


def train(args):
    cfg = read_simple_yolo_yaml(args.data)
    train_paths = image_paths(split_image_dir(cfg, "train"), args.max_train)
    val_paths = image_paths(split_image_dir(cfg, "valid"), args.max_val)

    train_ds = AerialYoloSeqDataset(
        train_paths,
        image_size=args.image_size,
        seq_len=args.seq_len,
        box_mode=args.box_mode,
        fps=args.fps,
        augment=not args.no_augment,
    )
    val_ds = AerialYoloSeqDataset(
        val_paths,
        image_size=args.image_size,
        seq_len=args.seq_len,
        box_mode=args.box_mode,
        fps=args.fps,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model = CfcAerialDetector(args.image_size, args.feat_dim, args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Device: {device}")
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    print(f"Image size: {args.image_size}")
    print(f"Seq len: {args.seq_len}")

    best_recall = -1.0
    for epoch in range(1, args.epochs + 1):
        stats = train_one_epoch(model, train_loader, opt, device, args.box_weight)
        precision, recall = evaluate(model, val_loader, device, th=args.th)
        print(
            f"epoch={epoch} loss={stats['loss']:.4f} obj={stats['obj']:.4f} "
            f"box={stats['box']:.4f} precision={precision:.3f} recall={recall:.3f}"
        )
        if recall >= best_recall:
            best_recall = recall
            save_checkpoint(args.model, model, args, epoch, precision, recall)
            print(f"Saved best model: {args.model}")


def save_checkpoint(path, model, args, epoch, precision, recall):
    ckpt = {
        "model": model.state_dict(),
        "image_size": args.image_size,
        "seq_len": args.seq_len,
        "feat_dim": args.feat_dim,
        "hidden": args.hidden,
        "box_mode": args.box_mode,
        "epoch": epoch,
        "precision": precision,
        "recall": recall,
        "arch": "rgb-cnn-cfc-aerial-v1",
    }
    torch.save(ckpt, args.model)


@torch.no_grad()
def predict(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    ckpt = load_checkpoint(args.model, device)
    model = CfcAerialDetector(ckpt["image_size"], ckpt["feat_dim"], ckpt["hidden"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    paths = image_paths(args.source, args.max_images)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = AerialYoloSeqDataset(paths, image_size=ckpt["image_size"], seq_len=ckpt["seq_len"])

    for i in tqdm(range(len(ds)), desc="predict"):
        x, dt, _obj, _box = ds[i]
        obj_logit, box = model(x.unsqueeze(0).to(device), dt.unsqueeze(0).to(device))
        prob = torch.sigmoid(obj_logit)[0].item()
        src_path = paths[i + ckpt["seq_len"] - 1]
        frame = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        if prob >= args.th:
            h, w = frame.shape[:2]
            cx, cy, bw, bh = box[0].cpu().numpy()
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"CfC human prob: {prob:.2f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        cv2.imwrite(str(out_dir / src_path.name), frame)
    print(f"Saved predictions: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare-okutama")
    p.add_argument("--source", type=str, required=True)
    p.add_argument("--video", type=str, default=None)
    p.add_argument("--labels", type=str, default=None)
    p.add_argument("--out", type=str, default="data/okutama_cfc")
    p.add_argument("--every", type=int, default=3)
    p.add_argument("--max-frames", type=int, default=900)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--no-generated", action="store_true")
    p.set_defaults(func=prepare_okutama)

    p = sub.add_parser("train")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--model", type=str, default="cfc_aerial_sard.pt")
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--seq-len", type=int, default=1)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--feat-dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--box-weight", type=float, default=10.0)
    p.add_argument("--box-mode", choices=("largest", "union"), default="union")
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--th", type=float, default=0.5)
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-val", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=train)

    p = sub.add_parser("predict")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--source", type=str, required=True)
    p.add_argument("--out", type=str, default="outputs/cfc_aerial_predict")
    p.add_argument("--th", type=float, default=0.5)
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

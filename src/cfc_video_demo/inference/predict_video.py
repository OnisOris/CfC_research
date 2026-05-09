from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from cfc_video_demo.inference.visualize import draw_box, draw_status
from cfc_video_demo.models.cnn_cfc_detector import CnnCfcDetector
from cfc_video_demo.utils.boxes import box_iou_cxcywh_np
from cfc_video_demo.utils.checkpoint import load_checkpoint
from cfc_video_demo.utils.video import infer_fps, transcode_h264_if_available, write_mp4


def window_dt(times: np.ndarray) -> np.ndarray:
    dt = np.diff(times, prepend=times[0])
    if len(dt) > 1:
        valid = dt[1:][dt[1:] > 1e-4]
        dt[0] = np.median(valid) if len(valid) else 1.0 / 30.0
    return np.clip(dt, 1e-4, 1.0).astype(np.float32)


@torch.no_grad()
def predict_sequence(args) -> None:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    ckpt = load_checkpoint(args.model, device)
    config = ckpt["config"]
    seq_len = args.seq_len if args.seq_len > 0 else int(config["seq_len"])

    model = CnnCfcDetector(
        image_size=int(config.get("image_size", 128)),
        feat_dim=int(config.get("feat_dim", 128)),
        hidden=int(config.get("hidden", 128)),
        spatial_pool=int(config.get("spatial_pool", 4)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    data = np.load(args.source, allow_pickle=False)
    frames = data["frames"]
    times = data["times"].astype(np.float32)
    labels_obj = data["labels_obj"].astype(np.float32) if "labels_obj" in data else None
    labels_box = data["labels_box"].astype(np.float32) if "labels_box" in data else None

    fps = args.fps if args.fps > 0 else infer_fps(times)
    out_frames = []
    scale = args.scale

    for idx in tqdm(range(len(frames)), desc="predict"):
        frame_rgb = frames[idx]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if scale != 1:
            frame_bgr = cv2.resize(frame_bgr, (frame_bgr.shape[1] * scale, frame_bgr.shape[0] * scale))

        prob = None
        pred_box = None
        iou = None
        if idx + 1 >= seq_len:
            start = idx + 1 - seq_len
            x_np = frames[start : idx + 1].astype(np.float32) / 255.0
            x_np = x_np.transpose(0, 3, 1, 2)
            x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(device)
            dt_np = window_dt(times[start : idx + 1])
            dt = torch.tensor(dt_np, dtype=torch.float32).unsqueeze(0).to(device)
            obj_logit, box = model(x, dt)
            prob = float(torch.sigmoid(obj_logit)[0].item())
            pred_box = box[0].detach().cpu().numpy()

            if prob > args.th:
                draw_box(frame_bgr, pred_box, (0, 0, 255), "pred", thickness=2)

        if labels_obj is not None and labels_box is not None and labels_obj[idx] > 0.5:
            gt_box = labels_box[idx]
            draw_box(frame_bgr, gt_box, (0, 255, 0), "gt", thickness=1)
            if pred_box is not None:
                iou = box_iou_cxcywh_np(pred_box, gt_box)

        draw_status(frame_bgr, prob, iou)
        out_frames.append(frame_bgr)

    write_mp4(out_frames, args.out, fps=fps)
    if args.h264:
        transcode_h264_if_available(args.out)
    print(f"Saved video: {args.out}")
    print(f"Frames: {len(out_frames)}")
    print(f"FPS: {fps:.2f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize CNN+CfC predictions on a prepared Caltech sequence.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--th", type=float, default=0.5)
    parser.add_argument("--seq-len", type=int, default=0)
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--h264", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    predict_sequence(args)


if __name__ == "__main__":
    main()

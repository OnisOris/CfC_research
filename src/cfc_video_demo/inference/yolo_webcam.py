from __future__ import annotations

import argparse
import os
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np
import torch

from cfc_video_demo.datasets.yolo_cache import load_yolo, top_person_feature
from cfc_video_demo.inference.visualize import draw_box
from cfc_video_demo.models.cnn_cfc_detector import YoloCfcRefiner
from cfc_video_demo.utils.checkpoint import load_checkpoint
from cfc_video_demo.utils.video import transcode_h264_if_available


def parse_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def feature_dt(times: list[float]) -> np.ndarray:
    arr = np.asarray(times, dtype=np.float32)
    dt = np.diff(arr, prepend=arr[0])
    if len(dt) > 1:
        valid = dt[1:][dt[1:] > 1e-4]
        dt[0] = np.median(valid) if len(valid) else 1.0 / 30.0
    return np.clip(dt, 1e-4, 1.0).astype(np.float32)


def open_writer(path: str | None, fps: float, frame_shape: tuple[int, int, int]) -> cv2.VideoWriter | None:
    if not path:
        return None
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    h, w = frame_shape[:2]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {out}")
    return writer


def draw_text(frame_bgr: np.ndarray, text: str, y: int, color: tuple[int, int, int]) -> None:
    cv2.putText(frame_bgr, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


@torch.no_grad()
def run_webcam(args) -> None:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    use_amp = args.amp and device == "cuda"
    ckpt = load_checkpoint(args.model, device)
    config = ckpt["config"]
    seq_len = args.seq_len if args.seq_len > 0 else int(config["seq_len"])

    refiner = YoloCfcRefiner(
        input_size=5,
        hidden=int(config.get("hidden", 96)),
        direct_weight=float(config.get("direct_weight", 0.25)),
    ).to(device)
    refiner.load_state_dict(ckpt["model"])
    refiner.eval()
    yolo = load_yolo(args.yolo_model)

    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3 or np.isnan(fps):
        fps = args.fps

    feature_window: deque[np.ndarray] = deque(maxlen=seq_len)
    time_window: deque[float] = deque(maxlen=seq_len)
    writer: cv2.VideoWriter | None = None
    start_time = time.monotonic()
    frame_idx = 0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            now = time.monotonic() - start_time
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = yolo.predict(
                source=[frame_rgb],
                imgsz=args.imgsz,
                conf=args.yolo_conf,
                iou=args.yolo_iou,
                classes=[0],
                device=args.device or None,
                verbose=False,
            )[0]
            feature = top_person_feature(result)
            feature_window.append(feature)
            time_window.append(now)

            raw_conf = float(feature[0])
            raw_box = feature[1:5]
            if raw_conf >= args.yolo_conf and raw_box[2] > 0 and raw_box[3] > 0:
                draw_box(frame_bgr, raw_box, (255, 0, 0), f"YOLO {raw_conf:.2f}", thickness=1)

            refined_prob = None
            if len(feature_window) == seq_len:
                x_np = np.stack(feature_window).astype(np.float32)
                dt_np = feature_dt(list(time_window))
                x = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0)
                dt = torch.tensor(dt_np, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    obj_logit, box = refiner(x, dt)
                refined_prob = float(torch.sigmoid(obj_logit)[0].detach().cpu().item())
                refined_box = box[0].detach().cpu().numpy()
                if refined_prob >= args.th:
                    draw_box(frame_bgr, refined_box, (0, 0, 255), f"CfC {refined_prob:.2f}", thickness=2)

            status = f"CfC warming {len(feature_window)}/{seq_len}" if refined_prob is None else f"CfC prob {refined_prob:.2f}"
            draw_text(frame_bgr, status, 24, (0, 0, 255))
            draw_text(frame_bgr, f"YOLO conf {raw_conf:.2f}", 50, (255, 0, 0))

            if writer is None:
                writer = open_writer(args.out, fps, frame_bgr.shape)
            if writer is not None:
                writer.write(frame_bgr)

            if not args.no_display:
                cv2.imshow("YOLO + CfC refiner", frame_bgr)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

            frame_idx += 1
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    if args.out and args.h264:
        transcode_h264_if_available(args.out)
    if args.out:
        print(f"Saved video: {args.out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run YOLO + CfC temporal refiner on a webcam or video source.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt")
    parser.add_argument("--th", type=float, default=0.6)
    parser.add_argument("--yolo-conf", type=float, default=0.05)
    parser.add_argument("--yolo-iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seq-len", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--h264", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    run_webcam(parser.parse_args())


if __name__ == "__main__":
    main()

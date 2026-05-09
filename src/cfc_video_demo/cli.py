"""CLI entrypoints for the Caltech CfC video pipeline."""

from cfc_video_demo.datasets.caltech_convert import main as caltech_convert_main
from cfc_video_demo.datasets.yolo_cache import main as yolo_cache_main
from cfc_video_demo.inference.predict_video import main as caltech_predict_main
from cfc_video_demo.inference.yolo_webcam import main as yolo_webcam_main
from cfc_video_demo.training.eval import main as caltech_eval_main
from cfc_video_demo.training.train import main as caltech_train_main
from cfc_video_demo.training.train_yolo_refiner import eval_main as yolo_eval_main
from cfc_video_demo.training.train_yolo_refiner import train_main as yolo_train_main

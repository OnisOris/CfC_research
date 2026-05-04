# CfC research

## CfC aerial SAR person detection

I did not find a public downloadable LisaAlert/Beeline AI dataset. Public
articles describe an internal dataset collected from real search missions, but
do not provide an open download link.

For an open annotated search-and-rescue substitute, use SARD:
https://www.kaggle.com/datasets/nikolasgegenava/sard-search-and-rescue

It is already annotated with one class, `human`, and contains drone/SAR imagery.
The labels are stored in YOLO text format, but the training below uses the
repository's CfC-based aerial detector, not YOLO.
Download and unpack:

```bash
mkdir -p ~/Downloads data/sard

curl -L -o ~/Downloads/sard-search-and-rescue.zip \
  https://www.kaggle.com/api/v1/datasets/download/nikolasgegenava/sard-search-and-rescue

unzip -o ~/Downloads/sard-search-and-rescue.zip -d data/sard
```

Use a local dataset file because the exported Roboflow `data.yaml` has
paths that do not match this repository layout:

```yaml
path: /home/onis/code/CfC_research/data/sard/search-and-rescue
train: train/images
val: valid/images
test: test/images

names:
  0: human
```

Train the CfC aerial prototype:

```bash
uv run cfc-aerial-detector train \
  --data data/sard/sard.yaml \
  --model cfc_aerial_sard.pt \
  --image-size 640 \
  --seq-len 1 \
  --epochs 20 \
  --batch 8 \
  --workers 4
```

Run it on the SARD test split:

```bash
uv run cfc-aerial-detector predict \
  --model cfc_aerial_sard.pt \
  --source data/sard/search-and-rescue/test/images \
  --out outputs/cfc_aerial_sard_test \
  --th 0.5
```

The current CfC aerial detector is a research prototype: it predicts whether a
tile contains a person and emits one box per tile. When an image has multiple
people, labels are reduced to either a union box or the largest box. For true
high-resolution aerial video, split frames into overlapping tiles first, then
train/evaluate on tile sequences. The SARD export is already a 640x640
tiled/preprocessed dataset.

## Ready pedestrian datasets

The MOT17Det page is not reliable for downloading. For quick local CfC
experiments, use the small Kaggle Pedestrian Dataset instead:
https://www.kaggle.com/datasets/smeschke/pedestrian-dataset

It contains three pedestrian videos and CSV bounding boxes. The dataset page
lists it as CC0/Public Domain, so it is convenient for experiments.

Download it, then convert one video/CSV pair into this project's `.npz` format:

```bash
mkdir -p ~/Downloads data/pedestrian_kaggle

curl -L -o ~/Downloads/pedestrian-dataset.zip \
  https://www.kaggle.com/api/v1/datasets/download/smeschke/pedestrian-dataset

unzip -o ~/Downloads/pedestrian-dataset.zip -d data/pedestrian_kaggle

uv run cfc-motion-detector prepare-video-csv \
  --video data/pedestrian_kaggle/crosswalk \
  --csv data/pedestrian_kaggle/crosswalk.csv \
  --out-data data/crosswalk_person_seq.npz \
  --out-labels data/crosswalk_person_labels.npz
```

Then train with the converted labels:

```bash
uv run cfc-motion-detector train \
  --data data/crosswalk_person_seq.npz \
  --labels data/crosswalk_person_labels.npz \
  --model cfc_crosswalk_person.pt
```

Run the webcam demo. Keep the target area empty/still for the first couple of
seconds while it calibrates the live background:

```bash
uv run cfc-motion-detector demo \
  --model cfc_crosswalk_person.pt \
  --cam 0 \
  --th 0.5
```

This is only a quick pipeline check. The Kaggle videos have a pedestrian in
every frame, so the model does not learn a useful "no person" state and will
not generalize well to a close indoor webcam view. For a real webcam demo,
collect and annotate local camera data:

```bash
uv run cfc-motion-detector collect --seconds 60 --out data/webcam_room.npz

uv run cfc-motion-detector annotate \
  --data data/webcam_room.npz \
  --out data/webcam_room_labels.npz \
  --init-auto

uv run cfc-motion-detector train \
  --data data/webcam_room.npz \
  --labels data/webcam_room_labels.npz \
  --model cfc_webcam_room.pt
```

For a larger research dataset, use Caltech Pedestrians from CaltechDATA:
https://data.caltech.edu/records/f6rph-90m20

Caltech is much larger and also has pedestrian bounding-box annotations, but it
needs an additional Caltech annotation converter before training here.

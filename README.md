# CfC research

Small research demo for pedestrian localization in video with Closed-form
Continuous-time networks.

This is not YOLO. The model in this repository is:

```text
RGB frame sequence -> CNN encoder per frame -> CfC over time -> detection head
```

The detector predicts one target for the last frame of each sequence:

- `objectness`: whether a person is present
- `bbox`: normalized `cx cy w h`

For the first Caltech demo this is intentionally a single-target detector, not
a full multi-object detector. If a frame has multiple people, conversion uses
either the largest person box or the union of all person boxes.

## Caltech Pedestrian Pipeline

The primary dataset target is Caltech Pedestrians:
https://data.caltech.edu/records/f6rph-90m20

The dbcollection dataset page is useful for understanding the available
Caltech detection tasks and the original annotations:
https://dbcollection.readthedocs.io/en/latest/datasets/caltech_ped.html

### Download

```bash
mkdir -p data/caltech_pedestrians

curl -L -C - \
  -o data/caltech_pedestrians/data_and_labels.zip \
  "https://data.caltech.edu/records/f6rph-90m20/files/data_and_labels.zip?download=1"
```

### Unzip

```bash
unzip data/caltech_pedestrians/data_and_labels.zip -d data/caltech_pedestrians/raw
```

If the unzip creates `setXX.tar` files instead of extracted `setXX/*.seq`
directories, either unpack those tars manually or pass `--extract-tars` to the
converter.

### Convert

```bash
uv run cfc-caltech-convert \
  --raw-root data/caltech_pedestrians/raw \
  --out data/caltech_prepared \
  --image-size 128 \
  --frame-step 3 \
  --target-mode largest
```

Prepared data is written as one `.npz` per source video:

```text
data/caltech_prepared/
  train/
    set00_V000.npz
  val/
  test/
  manifest_train.jsonl
  manifest_val.jsonl
  manifest_test.jsonl
```

Each `.npz` contains resized RGB frames, frame times, objectness labels, and a
single normalized target box for each frame.

### Train

```bash
uv run cfc-caltech-train \
  --data data/caltech_prepared \
  --model outputs/cfc_caltech.pt \
  --seq-len 16 \
  --stride 4 \
  --epochs 20 \
  --batch 32
```

`seq_len` defaults to `16`; use at least `8` for meaningful temporal CfC
experiments.

### Eval

```bash
uv run cfc-caltech-eval \
  --data data/caltech_prepared \
  --model outputs/cfc_caltech.pt \
  --split val
```

The first demo reports precision, recall, F1, mean IoU on positive frames, and
recall at IoU 0.5. It does not compute mAP.

### Predict

```bash
uv run cfc-caltech-predict \
  --model outputs/cfc_caltech.pt \
  --source data/caltech_prepared/val/<some_sequence>.npz \
  --out outputs/cfc_caltech_demo.mp4
```

The visualization draws predicted boxes in red and ground-truth boxes in green.

## Smoke Test

After a small part of the dataset is available, use:

```bash
uv run cfc-caltech-convert \
  --raw-root data/caltech_pedestrians/raw \
  --out data/caltech_prepared_smoke \
  --image-size 128 \
  --frame-step 10 \
  --max-sequences 2

uv run cfc-caltech-train \
  --data data/caltech_prepared_smoke \
  --model outputs/cfc_caltech_smoke.pt \
  --seq-len 8 \
  --stride 2 \
  --epochs 1 \
  --batch 4 \
  --max-train-windows 128 \
  --max-val-windows 64

uv run cfc-caltech-eval \
  --data data/caltech_prepared_smoke \
  --model outputs/cfc_caltech_smoke.pt \
  --split val

uv run cfc-caltech-predict \
  --model outputs/cfc_caltech_smoke.pt \
  --source data/caltech_prepared_smoke/val/<some_sequence>.npz \
  --out outputs/cfc_caltech_smoke.mp4
```

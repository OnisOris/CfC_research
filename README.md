# CfC research

Исследовательский пример для локализации пешехода на видео с помощью
Closed-form Continuous-time networks (CfC).

Модель обрабатывает короткое окно кадров и предсказывает наличие пешехода и
ограничивающую рамку для последнего кадра окна:

```text
последовательность RGB-кадров -> CNN-энкодер для каждого кадра -> CfC по времени -> detection head
```

На текущем этапе это одноцелевой детектор. Для каждого окна он предсказывает:

- `objectness`: есть ли пешеход в последнем кадре;
- `bbox`: нормализованную рамку в формате `cx cy w h`.

Если в кадре несколько пешеходов, при конвертации данных используется один
целевой бокс: либо самый крупный бокс пешехода, либо объединение всех боксов.
Режим выбирается параметром `--target-mode`.

## Pipeline для Caltech Pedestrians

Основной датасет для примера — Caltech Pedestrians:
https://data.caltech.edu/records/f6rph-90m20

Описание оригинальных задач и аннотаций Caltech также есть в документации
dbcollection:
https://dbcollection.readthedocs.io/en/latest/datasets/caltech_ped.html

### Скачивание

```bash
mkdir -p data/caltech_pedestrians

curl -L -C - \
  -o data/caltech_pedestrians/data_and_labels.zip \
  "https://data.caltech.edu/records/f6rph-90m20/files/data_and_labels.zip?download=1"
```

### Распаковка

```bash
unzip data/caltech_pedestrians/data_and_labels.zip -d data/caltech_pedestrians/raw
```

Если после распаковки внутри лежат `setXX.tar`, а не директории
`setXX/*.seq`, можно распаковать tar-файлы вручную или передать конвертеру
флаг `--extract-tars`.

### Конвертация

```bash
uv run cfc-caltech-convert \
  --raw-root data/caltech_pedestrians/raw \
  --out data/caltech_prepared \
  --image-size 128 \
  --frame-step 3 \
  --target-mode largest
```

После конвертации данные сохраняются по одному `.npz` файлу на исходное видео:

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

Каждый `.npz` содержит уменьшенные RGB-кадры, временные метки, objectness-метки
и один нормализованный целевой бокс для каждого кадра.

### Обучение

```bash
uv run cfc-caltech-train \
  --data data/caltech_prepared \
  --model outputs/cfc_caltech.pt \
  --seq-len 16 \
  --stride 4 \
  --epochs 20 \
  --batch 32
```

Параметр `--seq-len` задает длину временного окна. По умолчанию используется
`16`; для быстрых проверок можно уменьшить до `8`.

Параметр `--stride` задает шаг между соседними окнами. Чем меньше шаг, тем
больше обучающих примеров и тем дольше обучение.

Если PyTorch видит CUDA и не передан флаг `--cpu`, обучение автоматически
запускается на GPU.

### Оценка

```bash
uv run cfc-caltech-eval \
  --data data/caltech_prepared \
  --model outputs/cfc_caltech.pt \
  --split val
```

Скрипт выводит precision, recall, F1, mean IoU на положительных кадрах и
recall при IoU 0.5. mAP в этом примере не считается.

### Предсказание на видео

```bash
uv run cfc-caltech-predict \
  --model outputs/cfc_caltech.pt \
  --source data/caltech_prepared/val/<some_sequence>.npz \
  --out outputs/cfc_caltech_demo.mp4
```

Визуализация рисует предсказанные рамки красным цветом, а разметку — зеленым.

## Быстрая проверка

Для проверки пайплайна на небольшой части данных:

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

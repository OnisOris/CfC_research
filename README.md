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

## Краткий итог исследования

В проекте проверялись несколько способов использовать CfC для локализации
человека на видео.

Главный вывод:

```text
CfC хорошо работает как временная модель поверх готовых числовых признаков.
CfC плохо работает как единственная модель, которая должна сама увидеть
человека в сыром изображении и точно поставить bbox.
```

Практически лучший вариант сейчас:

```text
YOLO на каждом кадре находит человека -> CfC уточняет результат во времени
```

Почему так получилось:

- CNN+CfC с нуля научился примерно определять наличие человека, но плохо
  учил координаты рамки.
- Grid/YOLO-like head внутри собственной модели улучшал train-loss, но не
  переносился хорошо на validation/test.
- Pretrained YOLO уже умеет видеть людей, а CfC полезно добавляет временной
  контекст: меньше пропусков и выше итоговый F1.

### Что означают метрики

- `precision`: насколько можно доверять предсказаниям модели. Если модель
  сказала "человек есть", какая доля таких срабатываний правильная.
- `recall`: сколько реальных людей модель нашла. Высокий recall означает
  меньше пропусков.
- `F1`: баланс между precision и recall. Это удобная общая метрика, когда
  важны и ложные срабатывания, и пропуски.
- `mean_iou`: среднее совпадение предсказанной рамки с правильной рамкой.
  `IoU=1.0` значит полное совпадение, `IoU=0.0` значит рамки не пересеклись.
- `recall_at_iou_0_5`: доля реальных объектов, которые модель нашла рамкой
  с `IoU >= 0.5`. Для детекции bbox это самая важная метрика попадания рамкой.

### Результаты экспериментов

| Подход | Split | Precision | Recall | F1 | Mean IoU | Recall@IoU0.5 | Вывод |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| CNN+CfC single bbox | val | 0.357 | 1.000 | 0.526 | 0.016 | 0.0069 | Видит objectness, bbox почти не учится |
| CNN+CfC grid head | val | 0.237 | 0.915 | 0.377 | 0.038 | 0.0059 | Локализация всё еще слабая |
| CNN+CfC YOLO-like head | val | 0.209 | 1.000 | 0.346 | 0.013 | 0.0088 | Лучше не стало |
| YOLO baseline | val, th=0.6 | 0.954 | 0.306 | 0.463 | 0.283 | 0.215 | Очень точный, но много пропусков |
| YOLO+CfC refiner | val, th=0.6 | 0.641 | 0.521 | 0.575 | 0.270 | 0.285 | Лучший баланс на val |
| YOLO baseline | test, th=0.6 | 0.965 | 0.252 | 0.399 | 0.264 | 0.169 | Много пропусков на test |
| YOLO+CfC refiner | test, th=0.6 | 0.724 | 0.559 | 0.631 | 0.253 | 0.263 | Лучший итоговый результат |

Итоговый checkpoint:

```text
outputs/yolo_cfc_refiner.pt
```

Рекомендуемый threshold:

```text
--th 0.6
```

На `test` этот вариант дал:

```text
F1:              0.3991 -> 0.6311
recall:          0.2515 -> 0.5592
recall@IoU0.5:   0.1694 -> 0.2629
```

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

## YOLO + CfC temporal refiner

Лучший рабочий вариант в этом репозитории — использовать pretrained YOLO как
покадровый детектор, а CfC как временной refiner поверх последовательности
YOLO-признаков:

```text
кадр -> YOLO person bbox/conf -> последовательность [conf, cx, cy, w, h] -> CfC -> refined bbox/objectness
```

На `data/caltech_prepared_384`, checkpoint `outputs/yolo_cfc_refiner.pt`,
threshold `--th 0.6`:

```text
Caltech test:
YOLO baseline      F1=0.3991  recall=0.2515  recall@IoU0.5=0.1694
YOLO + CfC refiner F1=0.6311  recall=0.5592  recall@IoU0.5=0.2629
```

### Кэш YOLO-признаков

```bash
uv run cfc-yolo-cache \
  --data data/caltech_prepared_384 \
  --yolo-model yolov8n.pt \
  --splits train,val,test \
  --batch 64 \
  --device 0
```

Команда добавляет в подготовленные `.npz` массив `yolo_features` в формате
`conf cx cy w h`.

Параметры:

- `--data`: папка с подготовленным Caltech-датасетом.
- `--yolo-model`: pretrained YOLO checkpoint. `yolov8n.pt` быстрый и легкий.
- `--splits`: какие части датасета обработать: `train,val,test`.
- `--batch`: сколько кадров YOLO обрабатывает за один раз. Больше быстрее,
  но требует больше видеопамяти.
- `--device`: устройство для YOLO. Обычно `0` для первой CUDA-видеокарты.

### Обучение refiner

```bash
uv run cfc-yolo-train \
  --data data/caltech_prepared_384 \
  --model outputs/yolo_cfc_refiner.pt \
  --history outputs/yolo_cfc_refiner_history.csv \
  --seq-len 16 \
  --stride 4 \
  --epochs 40 \
  --batch 256 \
  --hidden 96 \
  --lr 3e-4 \
  --box-weight 20 \
  --iou-weight 1.0 \
  --workers 4 \
  --amp
```

Параметры:

- `--seq-len`: сколько последних кадров смотрит CfC. В экспериментах
  использовалось `16`.
- `--stride`: шаг между обучающими окнами. `4` быстрее, `1-2` дает больше
  окон, но дольше обучается.
- `--epochs`: сколько проходов по обучающим данным.
- `--batch`: сколько окон обрабатывается за один шаг обучения.
- `--hidden`: размер скрытого состояния CfC. Больше может быть мощнее, но
  медленнее.
- `--lr`: learning rate, то есть размер шага оптимизации.
- `--box-weight`: насколько сильно штрафовать ошибку bbox.
- `--iou-weight`: насколько сильно штрафовать плохое пересечение рамок.
- `--amp`: mixed precision на CUDA; обычно ускоряет обучение.

### Оценка

```bash
uv run cfc-yolo-eval \
  --data data/caltech_prepared_384 \
  --model outputs/yolo_cfc_refiner.pt \
  --split test \
  --batch 512 \
  --th 0.6 \
  --amp
```

Параметры:

- `--split`: какую часть датасета оценивать: `val` или `test`.
- `--th`: threshold objectness. Больше threshold — меньше ложных срабатываний,
  но больше пропусков.
- `--batch`: размер batch для оценки.

По sweep на validation лучший общий баланс был на `--th 0.6`:

```text
th=0.50  precision=0.526  recall=0.574  F1=0.549  recall@IoU0.5=0.300
th=0.60  precision=0.641  recall=0.521  F1=0.575  recall@IoU0.5=0.285
th=0.70  precision=0.756  recall=0.456  F1=0.569  recall@IoU0.5=0.279
th=0.80  precision=0.885  recall=0.385  F1=0.537  recall@IoU0.5=0.262
```

### Вебкамера

```bash
uv run cfc-yolo-webcam \
  --model outputs/yolo_cfc_refiner.pt \
  --source 0 \
  --yolo-model yolov8n.pt \
  --th 0.6 \
  --device 0 \
  --amp
```

Параметры:

- `--source`: источник видео. `0` — первая вебкамера, `1` — вторая, либо путь
  к видеофайлу.
- `--th`: threshold для красной refined-рамки CfC. Рекомендуется `0.6`.
- `--yolo-conf`: минимальная уверенность YOLO для синей сырой рамки.
- `--device`: CUDA-устройство для YOLO, обычно `0`.
- `--out`: путь для записи результата в mp4.
- `--no-display`: не открывать окно, только писать видео.

Важно: текущая вебкамера работает в single-target режиме. Она берет самого
уверенного человека от YOLO и уточняет только его. Для нескольких людей нужен
tracker и отдельное CfC-окно на каждый track.

Закрыть окно можно клавишей `q` или `Esc`. Для записи без окна:

```bash
uv run cfc-yolo-webcam \
  --model outputs/yolo_cfc_refiner.pt \
  --source 0 \
  --out outputs/webcam_yolo_cfc.mp4 \
  --no-display \
  --th 0.6 \
  --device 0 \
  --amp
```

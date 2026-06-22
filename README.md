# BG_ResearchWork

Курсовой проект по персонализированной генерации лиц на базе **замороженной diffusion-модели**.

## Идея

Используется связка:
- **SD1.5** как frozen backbone (VAE/UNet/Text Encoder не дообучаются полностью)
- **Arc2Face-путь для ID**: embedding лица извлекается из исходного фото, не обучается и встраивается в prompt-эмбеддинги
- **Отдельная обучаемая hair-ветка**: сегментация волос (BiSeNet) -> CLIP Vision -> projection в cross-attention

Цель: генерировать лицо с сохранением identity и контролем прически.

## Текущий статус

- Identity-часть frozen: ArcFace/Arc2Face используется только для text stream
- Hair-ветка trainable (`hair_cond`, `hair_proj`)
- В UNet внедрен `DualImageAttnProcessor` во все `attn2` блоки
- Добавлены диагностические режимы:
  - `hair_on`: Arc2Face text + hair
  - `hair_off`: Arc2Face text + 0
  - `empty_text_hair`: empty ArcFace text + hair
  - `empty_text_off`: empty ArcFace text + 0
  - `cross_hair`: Arc2Face text(A) + hair(B)

## Структура проекта

- `train.py` — обучение
- `inference.py` — инференс по CSV-парам
- `metrics.py` — метрики (ID/hair/FID)
- `config.yaml` — основной конфиг
- `src/model/id_conditioner_insightface.py` — frozen ArcFace extractor for Arc2Face text conditioning
- `src/model/hair_conditioner_parsing.py` — hair conditioning + BiSeNet маски
- `src/model/dual_ip_attention.py` — dual conditioning в cross-attention
- `src/utils/project_face_embs.py` — Arc2Face-проекция эмбеддингов лица в text stream

## Подготовка окружения (Colab)

Рекомендуемый базовый набор:

```bash
pip install -U diffusers transformers accelerate safetensors einops opencv-python tqdm pyyaml scikit-image
pip install insightface==0.7.3 onnxruntime-gpu
```

или:

```bash
# torch/torchvision поставь отдельно под CUDA кластера
pip install -r requirements.txt
```

Также нужны веса BiSeNet (face parsing):
- положить файл `79999_iter.pth` (или совместимый) и указать путь в `config.yaml`:
  - `models.hair_parsing_weights`

И модели InsightFace `antelopev2` должны быть доступны в `./models/antelopev2`.
При необходимости можно явно задать root:
- через конфиг `models.insightface_root`
- или переменной окружения `INSIGHTFACE_MODEL_ROOT`

Ожидаемая структура:

```text
<INSIGHTFACE_MODEL_ROOT>/models/antelopev2/*.onnx
```

## Данные

`train.py` ожидает папки с изображениями:

- `data.train_dir`
- `data.val_dir`

Датасет читается через `ImageFolderDataset` (`src/data/images.py`).

### Честный identity-disjoint paired protocol

Для проверки переноса волос лучше не делить картинки случайно: один и тот же человек
может встретиться на нескольких фото и попасть одновременно в train и validation.
Для этого есть протокол на ArcFace-кластерах:

```bash
python scripts/build_identity_splits.py \
  --image-root /path/to/FFHQ_clean_nohat_nohairmiss \
  --out-dir data/ffhq_identity_split \
  --insightface-root /path/to/insightface_root \
  --device cuda \
  --eps 0.5 \
  --min-samples 4 \
  --min-cluster-size 2 \
  --val-frac 0.1
```

Скрипт сохраняет:
- `identity_manifest.csv` — путь, ArcFace/DBSCAN cluster id, split
- `train_pairs.csv` — пары для обучения
- `val_pairs.csv` — пары для validation
- `split_summary.json` — сводка по кластерам и split

В paired CSV:
- `target` / `ref_id` — картинка target и identity condition
- `ref_hair` — другая картинка для hair condition
- `target_cluster` и `hair_cluster` различаются, чтобы hair condition не был тем же identity

Чтобы включить этот режим:

```yaml
data:
  train_pairs_csv: data/ffhq_identity_split/train_pairs.csv
  val_pairs_csv: data/ffhq_identity_split/val_pairs.csv
```

Если `train_pairs_csv` не задан, сохраняется старый режим чтения из `data.train_dir`.

## Обучение

```bash
python train.py --cfg config.yaml
```

Для кластера удобно сделать отдельный конфиг:

```bash
cp config.cluster.example.yaml config.cluster.yaml
# затем поправить пути под кластер
python train.py --cfg config.cluster.yaml
```

Артефакты сохраняются в:
- `runs/<exp_name>/samples/*.png` — qualitative rows
- `runs/<exp_name>/hair_debug/*.png` — `orig | hair_mask | hair_masked`
- `runs/<exp_name>/ckpt_step*.pt` — чекпоинты

## Интерпретация qualitative row

Текущий формат строки:

`original | hair_on | hair_off | empty_text_hair | empty_text_off | cross_hair | hair_source_B | hair_source_B_masked`

Где:
- `hair_on` должно быть основным результатом: Arc2Face text + hair condition
- `hair_off` показывает базовое Arc2Face-воспроизведение без hair condition
- `empty_text_hair` показывает вклад hair condition без ArcFace identity text
- `empty_text_off` показывает полностью выключенные дополнительные условия
- `cross_hair` нужен для проверки переноса волос из другого источника B

## Важные train-параметры

### Базовые
- `train.batch_size`
- `train.lr`
- `train.dual_lr_mult`
- `train.hair_aux_weight`

### Cross-hair (влияние source B)
- `train.cross_hair_clip_weight`
- `train.cross_hair_clip_every`
- `train.cross_hair_clip_batch`
- `train.cross_hair_decode_size`

### Скорость / profiling
- `train.cache_arcface_embs` — кеш ArcFace embedding по `path` в RAM (ускоряет CPU bottleneck на face extraction)
- `train.arcface_cache_max_items` — лимит элементов кеша
- `train.profile_timing` — включает timing по блокам training loop
- `train.profile_every` — как часто печатать `[timing avg/...]`

#### Быстрые пресеты

Выключить влияние B полностью:

```yaml
train:
  cross_hair_clip_weight: 0.0
```

Слабое влияние B:

```yaml
train:
  cross_hair_clip_weight: 0.1
  cross_hair_clip_every: 8
```

Сильнее влияние B:

```yaml
train:
  cross_hair_clip_weight: 0.25
  cross_hair_clip_every: 2
```

## Инференс

Подготовить `pairs.csv` с колонками:
- `pair_id,ref_id,ref_hair`

После запуска `inference.py` структура выходов такая:
- `out_dir/<pair_id>/gen.png`
- `out_dir/<pair_id>/ref_id.png`
- `out_dir/<pair_id>/ref_hair.png`
- `out_dir/manifest.csv`

Запуск:

```bash
python inference.py \
  --pairs_csv /path/to/pairs.csv \
  --out_dir /path/to/infer_out \
  --sd_model_id runwayml/stable-diffusion-v1-5 \
  --arc2face_repo_id FoivosPar/Arc2Face \
  --clip_vision_id openai/clip-vit-large-patch14 \
  --hair_weights /path/to/79999_iter.pth \
  --ckpt /path/to/ckpt_stepXXXX.pt \
  --insightface_root /path/to/insightface_root \
  --scale_hair 0.65 \
  --hair_class 17 \
  --hair_classes 17 \
  --hair_mask_dilate_kernel 1 \
  --hair_mask_dilate_iters 1 \
  --hair_focus_crop 1 \
  --hair_focus_crop_margin 0.20 \
  --hair_focus_crop_square 1
```

### Обязательный контроль: no-hair на step 0

Перед сравнением обученных чекпоинтов нужно запустить baseline:

- `ckpt_step0.pt`
- `scale_hair=0.0`
- те же `pairs.csv`, `seed`, `steps`, `guidance`, что и для финального чекпоинта

Если уже на `ckpt_step0 + scale_hair=0.0` генерации выглядят сломанными, проблема не в
обучении hair-ветки, а в базовом воспроизведении Arc2Face/инференса.

SLURM-шаблон:

```bash
PAIRS_CSV=eval/ckpt_compare/pairs_256.csv \
OUT_DIR=runs/step0_nohair \
CKPT_PATH=runs/<exp_name>/ckpt_step0.pt \
HAIR_WEIGHTS=/home/arobryadchikova/weights/79999_iter.pth \
SCALE_HAIR=0.0 \
sbatch scripts/infer.sbatch
```

Готовая ячейка для Colab:

```bash
!cd /content/BG_ResearchWork && python inference.py \
  --pairs_csv /content/pairs.csv \
  --out_dir /content/infer_out \
  --sd_model_id runwayml/stable-diffusion-v1-5 \
  --arc2face_repo_id FoivosPar/Arc2Face \
  --clip_vision_id openai/clip-vit-large-patch14 \
  --hair_weights /content/weights/79999_iter.pth \
  --ckpt /content/BG_ResearchWork/runs/method1_clip_shortcut/ckpt_step2000.pt \
  --steps 30 \
  --guidance 7.0 \
  --seed 123 \
  --scale_hair 0.65 \
  --hair_class 17 \
  --hair_classes 17 \
  --hair_mask_dilate_kernel 1 \
  --hair_mask_dilate_iters 1 \
  --hair_focus_crop 1 \
  --hair_focus_crop_margin 0.20 \
  --hair_focus_crop_square 1
```

## Метрики

`metrics.py` поддерживает:
- ID similarity (ArcFace cosine)
- Hair overlap (IoU / Dice)
- Hair perceptual distances (CLIP/DINO)
- FID / FID-CLIP

Пример запуска:

```bash
python metrics.py \
  --pairs_csv /path/to/pairs.csv \
  --gen_dir /path/to/infer_out \
  --hair_weights /path/to/79999_iter.pth \
  --insightface_root /path/to/insightface_root \
  --device cuda
```

По умолчанию `metrics.py` ищет генерации в формате:
- `--gen_pattern "{pair_id}/gen.png"`

Это уже совместимо с текущим `inference.py`.

Готовые ячейки для Colab:

```bash
!pip -q install -U scipy pandas
```

```bash
!cd /content/BG_ResearchWork && python metrics.py \
  --pairs_csv /content/pairs.csv \
  --gen_dir /content/infer_out \
  --hair_weights /content/weights/79999_iter.pth \
  --hair_class 17 \
  --device cuda \
  --seeds 123 \
  --out_csv /content/infer_out/metrics_results.csv
```

Опционально с FID/FID-CLIP:

```bash
!cd /content/BG_ResearchWork && python metrics.py \
  --pairs_csv /content/pairs.csv \
  --gen_dir /content/infer_out \
  --hair_weights /content/weights/79999_iter.pth \
  --hair_class 17 \
  --device cuda \
  --seeds 123 \
  --compute_fid 1 \
  --fid_preprocess face_parsing \
  --out_csv /content/infer_out/metrics_results_fid.csv
```

## Частые проблемы

1. **OOM на GPU**
- уменьшить `train.batch_size`
- увеличить `cross_hair_clip_every`
- уменьшить `cross_hair_clip_batch`
- оставить `cross_hair_decode_size=256`

2. **`cross_hair` почти не отличается от `hair_on`**
- увеличить `cross_hair_clip_weight`
- уменьшить `cross_hair_clip_every`
- затем контролировать артефакты (слишком сильный cross-loss может портить лицо)

## Запуск на кластере (SLURM)

В репозитории есть готовые шаблоны:
- `scripts/train.sbatch`
- `scripts/infer.sbatch`
- `scripts/metrics.sbatch`

Базовый сценарий:

```bash
cp config.cluster.example.yaml config.cluster.yaml
# отредактировать пути в config.cluster.yaml

python scripts/preflight.py --cfg config.cluster.yaml
mkdir -p logs
sbatch scripts/train.sbatch
```

Проверка статуса:

```bash
squeue -u $USER
tail -f logs/bg-train_<jobid>.out
```

## Примечание

Проект исследовательский: часть параметров требует ручного подбора в зависимости от датасета, GPU и цели эксперимента (ID-стабильность vs сила hair-transfer).

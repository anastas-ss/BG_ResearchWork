# BG_ResearchWork

Курсовой проект по персонализированной генерации лиц на базе **замороженной diffusion-модели**.

## Идея

Используется связка:
- **SD1.5** как frozen backbone (VAE/UNet/Text Encoder не дообучаются полностью)
- **Arc2Face-путь для ID**: embedding лица извлекается из исходного фото, не обучается и встраивается в prompt-эмбеддинги
- **Отдельная обучаемая hair-ветка**: сегментация волос (BiSeNet) -> CLIP Vision -> projection в cross-attention

Цель: генерировать лицо с сохранением identity и контролем прически.

## Текущий статус

- ID-ветка frozen (`trainable id_cond: 0`)
- Hair-ветка trainable (`hair_cond`, `hair_proj`)
- В UNet внедрен `DualImageAttnProcessor` во все `attn2` блоки
- Добавлены диагностические режимы:
  - `both_on`: text + id + hair
  - `id_only`: text + id + 0
  - `hair_only`: text + 0 + hair
  - `both_off`: text + 0 + 0
  - `cross_hair`: text + id(A) + hair(B)

## Структура проекта

- `train.py` — обучение
- `inference.py` — инференс по CSV-парам
- `metrics.py` — метрики (ID/hair/FID)
- `config.yaml` — основной конфиг
- `src/model/id_conditioner_insightface.py` — ArcFace/ID conditioning
- `src/model/hair_conditioner_parsing.py` — hair conditioning + BiSeNet маски
- `src/model/dual_ip_attention.py` — dual conditioning в cross-attention
- `src/utils/project_face_embs.py` — Arc2Face-проекция эмбеддингов лица в text stream

## Подготовка окружения (Colab)

Рекомендуемый базовый набор:

```bash
pip install -U diffusers transformers accelerate safetensors einops opencv-python tqdm pyyaml scikit-image
pip install insightface==0.7.3 onnxruntime-gpu
```

Также нужны веса BiSeNet (face parsing):
- положить файл `79999_iter.pth` (или совместимый) и указать путь в `config.yaml`:
  - `models.hair_parsing_weights`

И модели InsightFace `antelopev2` должны быть доступны в `./models/antelopev2`.

## Данные

`train.py` ожидает папки с изображениями:

- `data.train_dir`
- `data.val_dir`

Датасет читается через `ImageFolderDataset` (`src/data/images.py`).

## Обучение

```bash
python train.py --cfg config.yaml
```

Артефакты сохраняются в:
- `runs/<exp_name>/samples/*.png` — qualitative rows
- `runs/<exp_name>/hair_debug/*.png` — `orig | hair_mask | hair_masked`
- `runs/<exp_name>/ckpt_step*.pt` — чекпоинты

## Интерпретация qualitative row

Текущий формат строки:

`original | both_on | id_only | hair_only | both_off | cross_hair | hair_source_B | hair_source_B_masked`

Где:
- `both_on` должно быть основным результатом
- `id_only` показывает вклад ID без hair
- `hair_only` показывает вклад hair без ID
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
  --scale_id 0.0 \
  --scale_hair 1.0 \
  --hair_class 17
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
  --device cuda
```

## Частые проблемы

1. **OOM на GPU**
- уменьшить `train.batch_size`
- увеличить `cross_hair_clip_every`
- уменьшить `cross_hair_clip_batch`
- оставить `cross_hair_decode_size=256`

2. **Маска выделяет не волосы**
- проверить `cond.hair_class` (для BiSeNet обычно `17`)
- смотреть `hair_debug` картинки

3. **`cross_hair` почти не отличается от `both_on`**
- увеличить `cross_hair_clip_weight`
- уменьшить `cross_hair_clip_every`
- затем контролировать артефакты (слишком сильный cross-loss может портить лицо)

## Примечание

Проект исследовательский: часть параметров требует ручного подбора в зависимости от датасета, GPU и цели эксперимента (ID-стабильность vs сила hair-transfer).

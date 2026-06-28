# BG Research Work

Исследовательский проект по управляемому переносу прически в диффузионной генерации лиц. Идентичность задается через замороженный Arc2Face-путь, а признаки прически подключаются отдельной обучаемой веткой.

## Архитектура

- **Stable Diffusion v1.5** используется как замороженная базовая модель.
- **ArcFace + Arc2Face** формируют условие идентичности.
- **BiSeNet** выделяет область волос на изображении-референсе.
- **CLIP ViT-L/14 patch-токены** сохраняют пространственные признаки прически.
- **Decoupled cross-attention** подключает hair-токены отдельно от ID-условия.
- **Spatial gate** ограничивает влияние hair-ветки областью волос.
- **Независимый condition dropout** обучает режимы ID+hair, ID-only, hair-only и unconditional.
- Поддерживаются совместный и раздельный **classifier-free guidance**.

Базовые веса Stable Diffusion, Arc2Face, ArcFace, CLIP и BiSeNet не входят в репозиторий.

## Структура

```text
train.py                         обучение hair-ветки
inference.py                     генерация по CSV-парам
metrics.py                       ID- и hair-метрики
config.yaml                      базовый локальный конфиг
config.cluster.example.yaml      шаблон кластерного конфига
config.cluster.fullclean.long.yaml
                                 итоговый кластерный конфиг
src/data/images.py               датасеты и collate
src/model/hair_conditioner_parsing.py
                                 сегментация и CLIP patch-токены
src/model/dual_ip_attention.py   decoupled cross-attention
src/utils/project_face_embs.py   Arc2Face-совместимое ID-условие
scripts/build_identity_splits.py построение identity-disjoint split
scripts/convert_pairs_to_single_image.py
                                 перевод старых CSV в single-image train
scripts/train.sbatch             запуск обучения в SLURM
scripts/infer.sbatch             запуск инференса в SLURM
```

## Установка

Сначала установите PyTorch и torchvision под используемую версию CUDA, затем:

```bash
pip install -r requirements.txt
```

Дополнительно нужны:

1. веса BiSeNet face parsing, например `79999_iter.pth`;
2. модели InsightFace `antelopev2`;
3. доступ к моделям `runwayml/stable-diffusion-v1-5`, `FoivosPar/Arc2Face` и `openai/clip-vit-large-patch14`.

Ожидаемая структура InsightFace:

```text
<INSIGHTFACE_ROOT>/models/antelopev2/*.onnx
```

## Данные

При обучении используется одно изображение:

```text
target = ref_id = ref_hair
```

На инференсе `ref_id` и `ref_hair` могут указывать на разные изображения.

CSV должен содержать поля:

```text
pair_id,target,ref_id,ref_hair,target_cluster,hair_cluster,split
```

Identity-disjoint разбиение можно построить командой:

```bash
python scripts/build_identity_splits.py \
  --image-root /path/to/images \
  --out-dir data/ffhq_identity_split \
  --insightface-root /path/to/insightface \
  --device cuda \
  --eps 0.5 \
  --min-samples 4 \
  --min-cluster-size 2 \
  --val-frac 0.1
```

Для уже существующего разбиения:

```bash
python scripts/convert_pairs_to_single_image.py \
  --input-dir data/ffhq_identity_split \
  --out-dir data/ffhq_identity_split_single_image
```

Данные, эмбеддинги и сформированные CSV не коммитятся в репозиторий.

## Конфигурация

Основной кластерный конфиг:

```text
config.cluster.fullclean.long.yaml
```

Перед запуском укажите в нем пути к изображениям, CSV, весам BiSeNet и моделям InsightFace.

Ключевые параметры итоговой patch-архитектуры:

```yaml
cond:
  hair_token_mode: patch
  hair_max_patch_tokens: 64
  hair_patch_post_layernorm: true
  hair_patch_binary_mask: true
  hair_apply_token_mask_to_values: false
  hair_token_normalization: layernorm
  hair_spatial_gate: true
  hair_kv_init: text

train:
  id_cond_drop_prob: 0.15
  hair_cond_drop_prob: 0.15
  hair_localization_weight: 0.10
  hair_loss_inside_weight: 1.0
  hair_loss_outside_weight: 0.05
```

## Обучение

Локальный запуск:

```bash
python train.py --cfg config.cluster.fullclean.long.yaml
```

SLURM:

```bash
CFG_FILE=config.cluster.fullclean.long.yaml \
sbatch scripts/train.sbatch
```

Чекпоинты и промежуточные изображения сохраняются в `runs/<exp_name>/` и игнорируются Git.

## Инференс

Совместный CFG:

```bash
python inference.py \
  --pairs_csv eval/pairs.csv \
  --out_dir runs/inference \
  --sd_model_id runwayml/stable-diffusion-v1-5 \
  --arc2face_repo_id FoivosPar/Arc2Face \
  --clip_vision_id openai/clip-vit-large-patch14 \
  --insightface_root /path/to/insightface \
  --hair_weights /path/to/79999_iter.pth \
  --ckpt /path/to/ckpt_step9999.pt \
  --steps 25 \
  --guidance 3.0 \
  --scale_hair 0.25
```

Раздельный CFG:

```bash
python inference.py \
  --pairs_csv eval/pairs.csv \
  --out_dir runs/inference_factorized \
  --sd_model_id runwayml/stable-diffusion-v1-5 \
  --arc2face_repo_id FoivosPar/Arc2Face \
  --clip_vision_id openai/clip-vit-large-patch14 \
  --insightface_root /path/to/insightface \
  --hair_weights /path/to/79999_iter.pth \
  --ckpt /path/to/ckpt_step9999.pt \
  --cfg_mode factorized \
  --guidance_id 3.0 \
  --guidance_hair 1.0 \
  --scale_hair 0.25
```

Для SLURM те же параметры передаются переменными окружения в `scripts/infer.sbatch`.

## Метрики

```bash
python metrics.py \
  --pairs_csv eval/pairs.csv \
  --gen_dir runs/inference \
  --hair_weights /path/to/79999_iter.pth \
  --out_csv runs/inference/metrics.csv
```

Вычисляются `IDSim_arcface`, `Hair_IoU`, `Hair_Dice`, `dCLIP_hair`, `dDINO_hair` и опционально FID.

## Что не хранится в Git

В репозиторий не включаются датасеты, веса, чекпоинты, логи, результаты инференса, метрики, отчеты, изображения, материалы курсовой и вспомогательные отладочные скрипты.

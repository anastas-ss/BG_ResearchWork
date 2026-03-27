# Подробный отчет по инференсу и метрикам (job train=3795570, metrics=3795899)

## 1. Исходные данные
- Источник: архив `bg_report_3795570.zip` (распакован во временную директорию).
- Пар (samples): **64**
- Файл метрик: `/tmp/bg_report_3795570/bg_report_3795570/metrics_results.csv`
- Метрики: `IDSim_arcface`, `Hair_IoU`, `Hair_Dice`, `dCLIP_hair`, `dDINO_hair`.

## 2. Ключевые цифры
| Метрика | Mean | Median |
|---|---:|---:|
| IDSim_arcface | 0.059348 | 0.000000 |
| Hair_IoU | 0.277655 | 0.281205 |
| Hair_Dice | 0.403380 | 0.438969 |
| dCLIP_hair | 0.377103 | 0.358580 |
| dDINO_hair | n/a | n/a |

- `dDINO_hair` заполнено: **0/64** (в этом прогоне фактически отсутствует).
- Доля `ref_has_face=0`: **0.734375** (47/64)
- Доля `gen_has_face=0`: **0.703125** (45/64)
- `both_face=1`: **8/64**
- `none_face=1`: **36/64**

## 3. Группы по face detection
См. таблицу: `summary_by_face_group.csv`

## 4. Графики
1) Распределения метрик

![](figures/01_histograms_main_metrics.png)

2) Hair_IoU по группам детекции лиц

![](figures/02_boxplot_hair_iou_by_face_group.png)

3) IDSim vs Hair_IoU

![](figures/03_scatter_idsim_vs_hair_iou.png)

4) Mean vs Median

![](figures/04_mean_median_metrics.png)

5) Исходы детекции лиц

![](figures/05_face_detection_outcomes.png)

6) Корреляции

![](figures/06_correlation_heatmap.png)

## 5. Лучшие/худшие примеры
- `samples/top6_hair_iou_triplets.png`
- `samples/bottom6_hair_iou_triplets.png`

![](samples/top6_hair_iou_triplets.png)

![](samples/bottom6_hair_iou_triplets.png)

## 6. Дополнительные таблицы
- `top10_IDSim_arcface.csv`, `bottom10_IDSim_arcface.csv`
- `top10_Hair_IoU.csv`, `bottom10_Hair_IoU.csv`
- `top10_Hair_Dice.csv`, `bottom10_Hair_Dice.csv`
- `top10_dCLIP_hair.csv`, `bottom10_dCLIP_hair.csv`
- `summary_overall.csv`, `summary_by_face_group.csv`

## 7. Интерпретация
- Сегментационные hair-метрики (`Hair_IoU`, `Hair_Dice`) умеренные, но не высокие.
- `IDSim_arcface` в среднем низкий из-за большого числа кейсов без детекции лица в `gen`.
- Для части пар с успешной детекцией обеих сторон идентичность значительно выше (см. групповой summary).
- Для следующего отчета стоит добавить отдельный срез только по `both_face=1` как основной quality KPI.

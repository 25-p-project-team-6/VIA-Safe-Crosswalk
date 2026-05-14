# YOLO26n 7cls v2 comparison

Dataset: `/home/prml/StudentsWork/SeokJin/Pprojects/works_car_focus/03_Training/datasets/mixed_dataset_human_vehicle_signal_7cls_v2/data.yaml`

## Overall metrics

| Model | Metric | Value |
| --- | --- | ---: |
| YOLO11n 7cls v2 | precision | 0.82978 |
| YOLO11n 7cls v2 | recall | 0.73142 |
| YOLO11n 7cls v2 | mAP50 | 0.80106 |
| YOLO11n 7cls v2 | mAP50-95 | 0.54309 |
| YOLO26n 7cls v2 | precision | 0.85122 |
| YOLO26n 7cls v2 | recall | 0.73485 |
| YOLO26n 7cls v2 | mAP50 | 0.81527 |
| YOLO26n 7cls v2 | mAP50-95 | 0.55015 |

## Classwise mAP50-95 adoption gate

| Model | Metric | Value |
| --- | --- | ---: |
| YOLO11n 7cls v2 | human_green | 0.54200 |
| YOLO11n 7cls v2 | human_red | 0.56200 |
| YOLO11n 7cls v2 | vehicle_green | 0.46500 |
| YOLO11n 7cls v2 | vehicle_red | 0.39500 |
| YOLO26n 7cls v2 | human_green | 0.56559 |
| YOLO26n 7cls v2 | human_red | 0.56781 |
| YOLO26n 7cls v2 | vehicle_green | 0.50736 |
| YOLO26n 7cls v2 | vehicle_red | 0.43362 |

## Decision

adopt YOLO26n candidate for export review

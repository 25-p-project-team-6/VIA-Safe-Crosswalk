# Android App 전달용 모델 패키지: YOLO26n 7cls v2

## 요약

- 모델: **YOLO26n 기반 7-class v2 객체 검출 모델**
- 목적: 기존 YOLO11n 7cls v2 Android 모델 교체 후보
- 데이터셋: `mixed_dataset_human_vehicle_signal_7cls_v2`
- 학습 완료: 200 epoch full training
- 권장 적용 판단: **YOLO26n 7cls v2 사용 권장**
  - 전체 mAP50 상승
  - 전체 mAP50-95 상승
  - 핵심 클래스인 `human_green`, `human_red` 둘 다 상승
  - 차량 신호등 구분용 `vehicle_green`, `vehicle_red`도 상승

## 포함 파일

`models/` 아래 파일을 Android 앱에 선택 적용하면 됩니다.

| 파일 | 용도 | 입력 크기 | 출력 shape |
|---|---|---:|---|
| `best_yolo26n_7cls_v2_float16_640.tflite` | 정확도 우선 | 640 | `[1, 300, 6]` |
| `best_yolo26n_7cls_v2_float16_512.tflite` | 정확도/속도 절충 | 512 | `[1, 300, 6]` |
| `best_yolo26n_7cls_v2_float16_448.tflite` | 중간 옵션 | 448 | `[1, 300, 6]` |
| `best_yolo26n_7cls_v2_float16_416.tflite` | 중간 옵션 | 416 | `[1, 300, 6]` |
| `best_yolo26n_7cls_v2_float16_320.tflite` | 속도 우선 float16 | 320 | `[1, 300, 6]` |
| `best_yolo26n_7cls_v2_int8_640.tflite` | 경량/정확도 절충 int8 | 640 | `[1, 300, 6]` |
| `best_yolo26n_7cls_v2_int8_320.tflite` | 최경량/속도 우선 int8 | 320 | `[1, 300, 6]` |

기존 앱에서 사용하던 파일명 패턴과 구분되도록 `yolo26n_7cls_v2`를 파일명에 포함했습니다.

## 추천 적용 순서

1. 기존 앱이 float16 모델을 주로 쓰면 먼저 아래 파일로 교체 테스트
   - `best_yolo26n_7cls_v2_float16_640.tflite`
2. 모바일 성능이 부담되면 순서대로 낮은 입력 크기 테스트
   - `float16_512` → `float16_448` → `float16_416` → `float16_320`
3. 저사양/실시간 우선이면 int8 테스트
   - `best_yolo26n_7cls_v2_int8_320.tflite`
   - 단, int8는 실제 앱 환경에서 정확도/속도 재확인 필요

## 클래스 순서

앱 후처리에서 class id를 아래 순서로 해석해야 합니다. 기존 7cls v2와 동일합니다.

| id | class |
|---:|---|
| 0 | `bicycle` |
| 1 | `motorcycle` |
| 2 | `vehicle` |
| 3 | `human_green` |
| 4 | `human_red` |
| 5 | `vehicle_green` |
| 6 | `vehicle_red` |

## 입출력 / 전처리 / 후처리 메모

- 입력 shape: `[1, size, size, 3]`, NHWC
- 입력 dtype/정규화는 기존 Ultralytics TFLite 파이프라인과 동일하게 처리하면 됩니다.
  - 일반적으로 RGB 이미지 resize/letterbox 후 `0~1` 정규화 사용
- 출력 shape: `[1, 300, 6]`
- 기존 앱이 이전 7cls TFLite의 `[1, 300, 6]` 출력 파서를 사용 중이면 동일 계열로 연결하면 됩니다.
- 후처리 class id 매핑만 위 7-class 순서를 반드시 유지하세요.

## 성능 비교 요약

### 학습 `results.csv` 최고점 기준

| 항목 | 기존 YOLO11n | YOLO26n | 변화 |
|---|---:|---:|---:|
| 최고 mAP50 | 0.80156 | 0.81565 | +0.01409 |
| 최고 mAP50-95 | 0.54309 | 0.55015 | +0.00706 |
| final epoch mAP50 | 0.80123 | 0.81355 | +0.01232 |
| final epoch mAP50-95 | 0.54276 | 0.54861 | +0.00585 |

### 같은 validation 조건에서 best.pt 재검증 기준

| 항목 | YOLO11n best.pt | YOLO26n best.pt | 변화 |
|---|---:|---:|---:|
| Precision | 0.83041 | 0.85166 | +0.02125 |
| Recall | 0.73121 | 0.73491 | +0.00370 |
| mAP50 | 0.79732 | 0.81542 | +0.01810 |
| mAP50-95 | 0.53548 | 0.54961 | +0.01413 |

### 클래스별 mAP50-95

| class | YOLO11n | YOLO26n | 변화 |
|---|---:|---:|---:|
| bicycle | 0.45591 | 0.45602 | +0.00011 |
| motorcycle | 0.47010 | 0.48017 | +0.01008 |
| vehicle | 0.84690 | 0.83666 | -0.01024 |
| human_green | 0.54391 | 0.56559 | +0.02168 |
| human_red | 0.55692 | 0.56781 | +0.01089 |
| vehicle_green | 0.47432 | 0.50736 | +0.03304 |
| vehicle_red | 0.40030 | 0.43362 | +0.03332 |

## 앱 개발자 체크리스트

- [ ] 파일명을 앱 asset/model 로딩 설정에 반영
- [ ] 입력 크기별 resize/letterbox 설정 확인
- [ ] class id → label 매핑을 위 7-class 순서로 적용
- [ ] confidence threshold/NMS 정책 기존값으로 1차 테스트
- [ ] 사람 신호등(`human_green`, `human_red`)과 차량 신호등(`vehicle_green`, `vehicle_red`) 오검출 사례 위주로 샘플 검수
- [ ] 사용 모델별 FPS/지연시간 측정 후 최종 파일 선택

## 참고 경로

원본 학습 결과:

```text
yolo26n_human_vehicle_signal_7cls_v2/run_rebalanced_v2
```

원본 PyTorch best weight:

```text
yolo26n_human_vehicle_signal_7cls_v2/run_rebalanced_v2/weights/best.pt
```

TFLite export 원본 폴더:

```text
works_car_focus/03_Training/weights_yolo26n_7cls_v2_tflite
```


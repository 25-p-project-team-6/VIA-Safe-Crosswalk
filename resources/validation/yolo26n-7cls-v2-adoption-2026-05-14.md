# YOLO26n 7cls v2 Android 적용 기록 — 2026-05-14

## 관련 이슈
- GitHub Issue: #54
- 전달 패키지: `android_delivery_yolo26n_7cls_v2.zip`
- 보관 문서: `resources/models/yolo26n_7cls_v2/`

## 적용 판단
전달 handoff 기준으로 YOLO26n 7cls v2를 기존 YOLO11n 7cls v2 Android 모델의 교체 후보가 아니라 **현재 기본 후보군**으로 채택한다.

| 항목 | YOLO11n 7cls v2 | YOLO26n 7cls v2 | 변화 |
| --- | ---: | ---: | ---: |
| Precision | 0.82978 | 0.85122 | +0.02144 |
| Recall | 0.73142 | 0.73485 | +0.00343 |
| mAP50 | 0.80106 | 0.81527 | +0.01421 |
| mAP50-95 | 0.54309 | 0.55015 | +0.00706 |

핵심 신호 클래스도 모두 유지 또는 상승했다.

| class | YOLO11n | YOLO26n | 변화 |
| --- | ---: | ---: | ---: |
| human_green | 0.54200 | 0.56559 | +0.02359 |
| human_red | 0.56200 | 0.56781 | +0.00581 |
| vehicle_green | 0.46500 | 0.50736 | +0.04236 |
| vehicle_red | 0.39500 | 0.43362 | +0.03862 |

## 앱 반영 내용
- Android asset의 기존 `best_7cls_v2_*` TFLite 후보를 `best_yolo26n_7cls_v2_*` 후보로 교체한다.
- `DetectionLabels.modelFilesForActiveSchema()`는 YOLO26n 7cls 후보가 있으면 해당 파일만 모델 선택/온보딩 calibration에 노출한다.
- Class id 순서는 기존 7cls와 동일하므로 detector 후처리/label mapping은 변경하지 않는다.
- 출력 shape도 `[1, 300, 6]`로 기존 TFLite parser를 그대로 사용한다.

## 포함 모델
| 파일 | 입력 크기 | 용도 |
| --- | ---: | --- |
| `best_yolo26n_7cls_v2_float16_640.tflite` | 640 | 정확도 우선 |
| `best_yolo26n_7cls_v2_float16_512.tflite` | 512 | 정확도/속도 절충 |
| `best_yolo26n_7cls_v2_float16_448.tflite` | 448 | 중간 옵션 |
| `best_yolo26n_7cls_v2_float16_416.tflite` | 416 | 중간/속도형 |
| `best_yolo26n_7cls_v2_float16_320.tflite` | 320 | 속도 우선 float16 |
| `best_yolo26n_7cls_v2_int8_640.tflite` | 640 | 경량/정확도 절충 int8 |
| `best_yolo26n_7cls_v2_int8_320.tflite` | 320 | 최경량/속도 우선 int8 |

## 검증 체크리스트
- [ ] 온보딩 calibration에서 YOLO26n 후보만 측정되는지 확인
- [ ] 디버그 모델 spinner에서 YOLO26n 후보만 보이는지 확인
- [ ] S25+에서 기본 추천/선택 모델명 확인
- [ ] `human_green`/`human_red`와 `vehicle_green`/`vehicle_red` 분리 확인
- [ ] 차량 신호만 보이는 구간이 `GREEN_CONFIRMED`로 이어지지 않는지 확인
- [ ] 320/416 또는 448/512/640 후보별 FPS와 지연시간 비교

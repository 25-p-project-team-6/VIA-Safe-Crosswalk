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
- 출력 shape `[1, 300, 6]`은 NMS 포함 layout으로 보고 `score, classId` 컬럼을 분기 파싱한다.

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

## 후속 런타임 주의사항
- GitHub Issue #58에서 YOLO26n 640 float16 처리 FPS 저하를 별도로 추적한다.
- 전달된 YOLO26n TFLite 출력 `[1, 300, 6]`은 앱에서 `x1, y1, x2, y2, score, classId` 형태의 NMS 포함 export로 해석한다.
- 이 layout은 class id가 confidence처럼 표시되는 문제는 해결됐지만, Android TFLite GPU delegate에서는 NMS 포함 그래프가 raw-output export보다 느릴 수 있으므로 실측 비교가 필요하다.


## Raw-output follow-up — 2026-05-15
- Issue #58에서 확인된 640 FPS 저하 원인 분리를 위해 NMS-free / TopK-free raw-output export를 추가로 전달받았다.
- 현재 앱 assets는 `best_yolo26n_7cls_v2_raw_*` 파일을 사용하며, 기존 `[1, 300, 6]` NMS 포함 파일은 앱 assets에서 제거했다.
- Raw-output 640 float16은 `[1, 11, 8400]`으로 기존 YOLO11n 7cls v2와 같은 앱-side confidence/NMS 경로에서 비교한다.
- 실기기 로그에서 raw 640/GPU는 `layout=class_score_cxcywh`, `analysis=960x720`로 초기화되며, 초반 안정 구간에서 20fps대 처리까지 회복했다.
- 장시간/고온 상태에서는 Android thermal status 2에서 detect stage가 증가해 FPS가 다시 떨어질 수 있으므로, 이후 야외 검증에서는 모델명/해상도뿐 아니라 thermal state도 함께 기록한다.

## 검증 체크리스트
- [x] 온보딩 calibration에서 raw YOLO26n 후보를 우선 노출하도록 로직/테스트 확인
- [x] 디버그 모델 spinner에서 raw YOLO26n 후보를 우선 노출하도록 로직/테스트 확인
- [x] S25+에서 raw 640/GPU 선택 모델명과 `model_io` 확인
- [ ] `human_green`/`human_red`와 `vehicle_green`/`vehicle_red` 분리 확인
- [ ] 차량 신호만 보이는 구간이 `GREEN_CONFIRMED`로 이어지지 않는지 확인
- [x] raw 416/512/640 후보의 FPS와 지연시간 1차 비교

# 7클래스 모델 검증/튜닝 기록 — 2026-05-05

## 목적
Issue #40의 검증 기준을 앱 안에서 재현 가능하게 남긴다. 이 문서는 7클래스 모델을 리플레이/현장 영상으로 비교할 때 기록해야 할 항목, 현재 앱의 안전장치, 기본 추천 정책, threshold/NMS 점검 결과를 정리한다.

## 현재 앱에 포함된 7cls 후보
| 모델 파일 | 용도 | 비고 |
| --- | --- | --- |
| `best_7cls_v2_float16_640.tflite` | 정확도 우선 | GPU/고성능 기기 우선 후보 |
| `best_7cls_v2_float16_512.tflite` | 정확도-속도 중간 | 640이 15 FPS 미만일 때 비교 |
| `best_7cls_v2_float16_448.tflite` | 균형형 | 중간 해상도 비교 |
| `best_7cls_v2_float16_416.tflite` | 균형/속도형 | 640보다 빠른 후보 |
| `best_7cls_v2_float16_320.tflite` | 속도 우선 | 저성능/리플레이 빠른 점검 후보 |
| `best_7cls_v2_int8_640.tflite` | CPU/경량 비교 | INT8 calibration 품질 확인 필요 |
| `best_7cls_v2_int8_320.tflite` | CPU/저사양 비교 | 속도 우선 후보 |

`DetectionLabels.modelFilesForActiveSchema()`는 7cls 파일이 있으면 기존 `best_float16_*`/`best_int8_*` 모델을 목록에서 제외한다. 따라서 현재 앱 모델 선택 목록에는 7cls v2 후보만 노출되는 것이 정상이다.

## 현재 안전장치 확인
| 검증 항목 | 현재 앱 동작 | 근거 |
| --- | --- | --- |
| 보행자 신호 truth | `human_green`, `human_red`만 보행자 신호 색으로 매핑 | `DetectionLabels.pedestrianTrafficState()` |
| 차량 신호 억제 | `vehicle_green`, `vehicle_red`는 보행자 신호 색으로 쓰지 않고 `UNKNOWN` | `PostProcessor.observedTrafficLightState()` |
| 차량 신호만 보이는 경우 | `UNCERTAIN_VIEW` + 보행자 신호 확인 요청 | `SignalAdvisoryEvaluator` |
| 차량 신호가 함께 보이는 green | `GREEN_CONFIRMED` 차단 | `SignalAdvisoryEvaluator` tests |
| 7cls 모델 우선 목록 | 7cls 파일이 있으면 기존 모델 숨김 | `DetectionLabelsTest.activeSchemaPrefersSevenClassModelsWhenPresent` |
| NMS | 기본 IoU 0.5, 신호 클래스는 0.05로 더 엄격하게 중복 제거 | `YoloDetector` 생성부 |
| confidence threshold | 런타임은 신호등용 `trafficLightThreshold`와 일반 객체용 `generalObjThreshold`를 분리해 적용하고, calibration은 0.15 고정값으로 측정 | `MainActivity`, `OnboardingActivity` 호출부 |

## 기본 추천/튜닝 판단
현재 기본 정책은 **자동 측정 우선**이다.

1. 온보딩 calibration은 후보 모델을 측정한다.
2. 15 FPS 이상을 만족하는 후보 중 해상도가 가장 높은 모델을 선택한다.
3. 어떤 모델도 15 FPS를 만족하지 못하면 실제 측정 FPS가 가장 높은 모델을 선택한다.
4. 사용자는 디버그 패널에서 모델 파일명을 직접 선택해 320/416/448/512/640 및 float16/int8을 비교할 수 있다.

현재 코드 기준으로는 field/replay 혼동 사례가 더 쌓이기 전까지 traffic-light/general-object threshold와 NMS 기본값을 추가로 낮추거나 높이지 않는다. 특히 차량 신호가 보행자 신호로 이어지는 문제는 threshold보다 label 분리와 advisory gate에서 먼저 차단한다.

## 리플레이/현장 검증 기록표
각 영상마다 아래 표를 복사해 작성한다.

| 영상/장소 | 시간대 | 모델 | backend | 평균 processed FPS | human_red/green 확인 | vehicle_red/green 분리 | 차량→보행자 오검출 | 보행자→차량 오검출 | 기타 오검출 | 판단 |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| sample-1 | 주간/야간 | 640/416/320 등 | GPU/CPU |  |  |  |  |  |  |  |

## 혼동 케이스 분류
- **A. 차량 신호 → 보행자 신호 오검출**: 가장 위험. 발생 시 모델/threshold보다 advisory gate와 label truth 분리를 우선 확인한다.
- **B. 보행자 신호 → 차량 신호 오검출**: 보수적으로는 안내가 불확실해지는 방향이라 위험도는 A보다 낮지만 사용성 저하.
- **C. 신호등 아닌 물체 → 신호 오검출**: score, 크기, 중앙성, 지속시간 조건을 함께 확인한다.
- **D. 작은/가린 신호**: `needsZoomSuggestion` 또는 `UNCERTAIN_VIEW`로 빠지는지 확인한다.
- **E. 복수 신호**: `multipleSignalDetected`와 `UNCERTAIN_VIEW`가 나오는지 확인한다.

## 완료 판정 기준
#40을 닫으려면 최소한 다음을 채운다.

- 640 / 416 또는 448 / 320 중 3개 이상 모델 비교.
- float16 2개 이상, int8 1개 이상 비교.
- 실제 앱 리플레이 또는 현장 영상에서 pedestrian/vehicle signal 분리 확인.
- 차량용 신호만 보이는 구간이 보행자 `GREEN_CONFIRMED`로 이어지지 않음 확인.
- 기본 추천 모델 유지/변경 판단을 표에 기록.
- threshold/NMS 변경 필요 여부 기록.

## 현재 결론
- **코드상 안전장치**: 차량 신호는 보행자 truth로 쓰이지 않고, green advisory 확정도 차량 신호가 보이면 차단된다.
- **기본 모델 정책**: 자동 calibration 기반 선택을 유지한다.
- **threshold/NMS**: 현장 혼동 로그가 추가되기 전까지 기본값 유지.
- **남은 수동 검증**: 실제 샘플/현장 영상에서 모델별 FPS와 혼동 케이스 표를 채워야 최종 경험적 결론을 낼 수 있다.

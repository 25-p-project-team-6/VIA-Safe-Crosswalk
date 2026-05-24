# YOLO26n 7cls v2 raw-output Android 적용 기록 — 2026-05-15

## 관련 이슈
- GitHub Issue: #58
- 전달 패키지: `android_delivery_yolo26n_7cls_v2_raw_output.zip`
- 보관 문서: `resources/models/yolo26n_7cls_v2_raw_output/`

## 적용 목적
기존 YOLO26n Android delivery는 출력이 `[1, 300, 6]`인 NMS/TopK 포함 export였다. 이 형태는 class id/confidence 파싱 문제는 해결할 수 있었지만, S25+ 실시간 경로에서 640 float16 처리 FPS가 크게 낮았다.

이번 raw-output delivery는 `end2end=False`로 재-export된 NMS-free / TopK-free TFLite 세트이며, 기존 YOLO11n 7cls v2 Android 경로와 같은 조건으로 앱 내부 confidence/NMS를 수행하게 한다.

## Raw-output export 확인
Local flatbuffer metadata check:

| Candidate | Input | Output | Postprocess op clue |
| --- | --- | --- | --- |
| Previous YOLO11n 7cls v2 640 float16 | `[1, 640, 640, 3]` | `[1, 11, 8400]` | `TOPK_V2/GATHER_ND/TILE/FLOOR_MOD/GATHER = 0` |
| YOLO26n 7cls v2 raw 640 float16 | `[1, 640, 640, 3]` | `[1, 11, 8400]` | `TOPK_V2/GATHER_ND/TILE/FLOOR_MOD/GATHER = 0` |

`11 = 4 bbox + 7 class scores`이며 기존 `YoloOutputParser` class-score path가 처리한다.

## 앱 반영 내용
- App assets의 NMS-included `best_yolo26n_7cls_v2_*` 파일을 raw-output `best_yolo26n_7cls_v2_raw_*` 파일로 교체한다.
- `DetectionLabels.modelFilesForActiveSchema()`는 raw YOLO26n 파일이 있으면 raw 파일만 노출한다.
- 기존 non-raw YOLO26n 파일이 남아 있더라도 모델 spinner/온보딩 calibration이 raw 후보를 우선 사용하게 한다.
- 앱 기본 안전 fallback 파일명도 raw-output 320 float16 후보로 갱신한다.
- 2026-05-15 후속 튜닝에서 기본 정책은 raw 640 float16/GPU를 고해상도 후보로 유지하되, 실시간 처리 루프의 기본 최대 처리 FPS를 20fps로 제한해 발열과 장기 throttling을 줄이는 방향으로 정리했다. raw 512는 30fps 근처 비교/수동 후보로 남긴다.

## 포함 모델
| 파일 | 입력 크기 | 출력 shape | 용도 |
| --- | ---: | --- | --- |
| `best_yolo26n_7cls_v2_raw_float16_640.tflite` | 640 | `[1, 11, 8400]` | 기존 YOLO11n 640과 같은 조건의 우선 비교 후보 |
| `best_yolo26n_7cls_v2_raw_float16_512.tflite` | 512 | `[1, 11, 5376]` | 정확도/속도 절충 |
| `best_yolo26n_7cls_v2_raw_float16_448.tflite` | 448 | `[1, 11, 4116]` | 중간 옵션 |
| `best_yolo26n_7cls_v2_raw_float16_416.tflite` | 416 | `[1, 11, 3549]` | 중간/속도형 |
| `best_yolo26n_7cls_v2_raw_float16_320.tflite` | 320 | `[1, 11, 2100]` | 속도 우선 float16 |
| `best_yolo26n_7cls_v2_raw_int8_640.tflite` | 640 | `[1, 11, 8400]` | CPU/int8 640 비교 |
| `best_yolo26n_7cls_v2_raw_int8_320.tflite` | 320 | `[1, 11, 2100]` | CPU/int8 속도 baseline |

## 실기기 검증 계획
- raw 640 float16/GPU의 `model_io`가 `output=[1, 11, 8400] layout=class_score_cxcywh`로 뜨는지 확인한다.
- NMS-included 640 float16/GPU 측정치와 raw 640 float16/GPU 측정치를 같은 카메라 경로에서 비교한다.
- raw 640이 기존 YOLO11n 640에 가까운 처리 FPS로 회복되는지 확인한다.

## 실기기 검증 결과 — 2026-05-15
Debug APK를 설치한 뒤 raw-output 모델을 명시 선택해 라이브 카메라 경로에서 확인했다.

`best_yolo26n_7cls_v2_raw_float16_640.tflite` 초기화 로그:

```text
model_io model=best_yolo26n_7cls_v2_raw_float16_640.tflite backend=GPU requestedGpu=true compat=true input=[1, 640, 640, 3] output=[1, 11, 8400] rows=8400 cols=11 transposed=false layout=class_score_cxcywh labels=7
model=best_yolo26n_7cls_v2_raw_float16_640.tflite backend=GPU requestedGpu=true compat=true analysis=960x720
```

같은 세션에서 초반 안정 구간은 raw 640/GPU가 Camera FPS와 Processed FPS를 거의 같이 따라가며 20fps대까지 회복했다.

| Model | Backend | Analysis | Observed segment | Processed FPS | Avg latency / detect clue |
| --- | --- | --- | --- | ---: | --- |
| `best_yolo26n_7cls_v2_raw_float16_640.tflite` | GPU | 960x720 | warm, first stable window | 20.06..21.24 | avg latency ~40ms, detect stage ~43ms |
| `best_yolo26n_7cls_v2_raw_float16_640.tflite` | GPU | 960x720 | later thermal-throttled window | 9.3..12.3 | detect stage grows to ~90ms |
| `best_yolo26n_7cls_v2_raw_float16_512.tflite` | GPU | 768x576 | same warmed device state | 13.4..13.8 | detect stage ~59..62ms |
| `best_yolo26n_7cls_v2_raw_float16_416.tflite` | GPU | 640x480 | same warmed device state | 17.0..17.5 | detect stage ~46..48ms |

During the later drop, Android thermal service reported `Thermal Status: 2` and skin temperature status `2`. The FPS decline therefore should not be interpreted as the raw-output export still containing the old NMS/TopK bottleneck. The raw 640 export removes the `[1, 300, 6]` postprocess graph and restores the expected app-side NMS parser path; sustained FPS still depends on device thermal state.

## Input FPS label correction — 2026-05-15
The first live retest exposed a measurement bug: the debug UI's Camera FPS was being marked from the `ImageAnalysis` analyzer callback. Because the analyzer retained an `ImageProxy` until the slower processing executor copied and closed it, CameraX backpressure could make the "Camera FPS" label follow model latency instead of the actual camera capture cadence. Replay had the same class of bug: "Replay FPS" was marked in the processing loop rather than from rendered video-frame updates.

The debug input-rate source was changed as follows:

- Live camera input FPS is now marked from a Camera2 session capture callback on the preview stream.
- The analyzer callback remains a fallback until the first capture callback is observed.
- Replay input FPS is now marked from `TextureView.onSurfaceTextureUpdated`.
- `RateTracker` methods are synchronized because camera/replay input ticks can arrive off the UI thread.

Post-fix live log with raw 640/GPU under warmed/slow processing:

```text
input="Camera FPS: 29.82" processed="Processed FPS: 9.67"
input="Camera FPS: 29.85" processed="Processed FPS: 9.52"
input="Camera FPS: 29.98" processed="Processed FPS: 9.57"
```

This confirms the input label is no longer a proxy for inference throughput. The processed FPS can still drop with thermal/inference cost, but Camera FPS stays near the requested 30fps capture range.

## 20fps 처리 cap 정책 — 2026-05-15
현장 관찰상 raw 640 float16/GPU는 초반 안정 구간에서 대략 20fps대 처리, raw 512 float16/GPU는 30fps 근처 처리가 가능했다. 30fps를 계속 전부 처리하려고 밀어붙이면 발열/thermal throttling으로 오히려 장기 안정성이 떨어질 수 있으므로, 자동 추천은 고해상도 raw 640 후보를 유지하고 런타임 처리 루프에 기본 20fps cap을 둔다.

- GPU float16 자동 추천 target input은 640을 유지한다.
- 온보딩 calibration 통과 기준은 20fps다. 640이 20fps를 넘으면 고해상도 후보가 선택된다.
- Live camera capture는 Camera2 AE target range를 30fps로 유지하되, 분석 처리 루프는 기본 20fps로 최신 프레임만 샘플링한다.
- 디버그 패널에서 Max Processed FPS를 10..30fps 범위로 조정할 수 있다. Replay도 같은 cap으로 샘플링한다.
- Replay도 원본 영상이 60fps여도 앱 입력 목표 표시를 `Replay FPS: 30.00`으로 고정하고, 처리량은 `Processed FPS`에서 별도 확인한다.

## 로컬 검증
- `./gradlew testDebugUnitTest --tests 'kr.co.gachon.pproject6.via.ml.DetectionLabelsTest' --tests 'kr.co.gachon.pproject6.via.ml.InferenceModelProfileTest' --tests 'kr.co.gachon.pproject6.via.ml.YoloOutputParserTest'` ✅
- `./gradlew testDebugUnitTest` ✅
- `./gradlew lintDebug` ✅
- `./gradlew assembleDebug` ✅
- Debug APK install and live-camera log capture ✅
- Debug APK reinstall after input-FPS label fix; live-camera logs show Camera FPS ~30 while Processed FPS ~9..10 ✅

## 15fps thermal-stability policy update — 2026-05-24
S25+ field testing after the 24fps camera-input fallback showed that camera input held near 24fps, but sustained raw 640 float16/GPU processing dropped to roughly 11.6..12.0 Processed FPS as device temperature rose. To reduce oscillation and make onboarding recommendations match the demo operating point, the runtime policy is updated to prefer exact 15fps camera input when the HAL exposes it, set the default Max Processed FPS to 15, and lower onboarding calibration's pass target to 15fps.

Policy after this update:
- Live Camera2 target range prefers exact `15-15` for thermal/battery stability.
- If exact 15fps is unavailable, the app can fall back to the nearest fixed range above 15 rather than a wide variable range.
- Onboarding calibration target is 15fps, so model selection is evaluated against the same sustained demo target.
- Debug Camera FPS remains the real callback rate; no synthetic FPS label is used.

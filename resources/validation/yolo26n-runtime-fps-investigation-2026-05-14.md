# YOLO26n runtime FPS investigation — 2026-05-14

## Related issue
- GitHub Issue: #58
- Base branch: `fix/53-s25-red-led-flicker-artifact`
- Investigation branch: `investigate/58-yolo26n-runtime-fps`

## Symptom
After adopting YOLO26n 7cls v2, the 640 float16 candidate is noticeably slower than expected in the realtime Android path. The previous 7cls 640 float16 path had been calibrated around the mid-20 FPS range, while the YOLO26n 640 float16 path was observed around the high single-digit to low double-digit processed FPS range. Switching to the YOLO26n 320 float16 candidate recovers throughput.

This should not be treated as proof that YOLO26n itself is heavier. Runtime export layout and delegate compatibility can dominate Android TFLite latency.

## Evidence already available
From `resources/models/yolo26n_7cls_v2/manifest.txt`, every delivered YOLO26n TFLite candidate has the same output shape:

| Model family | Input | Output |
| --- | ---: | --- |
| YOLO26n 7cls v2 float16 | 320..640 | `[1, 300, 6]` |
| YOLO26n 7cls v2 int8 | 320/640 | `[1, 300, 6]` |

The app parser now treats this six-column seven-class output as an NMS-included layout:

```text
x1, y1, x2, y2, score, classId
```

That fixed the earlier symptom where class ids such as `2.00` and `6.00` were displayed as confidence scores. However, it also means the exported TFLite graph may contain detection post-processing/NMS operations instead of exposing raw YOLO class-score tensors to the app.

## Flatbuffer inspection snapshot
A local TFLite flatbuffer metadata check separates the previous raw-output export from the delivered YOLO26n export:

| Candidate | Output tensor | Operator count | Runtime-relevant operator clue |
| --- | --- | ---: | --- |
| Previous YOLO11n 7cls v2 640 float16 | `[1, 11, 8400]` | 546 | raw class-score tensor; no `TOPK_V2` in graph |
| YOLO26n 7cls v2 640 float16 | `[1, 300, 6]` | 685 | NMS-like tail includes `TOPK_V2`, `GATHER_ND`, `TILE`, `FLOOR_MOD` |

This confirms that the new Android delivery is not just a smaller backbone swap. It changes the TFLite output contract from raw predictions to a post-processed top-300 detection tensor, so runtime speed must be measured as an export/runtime property, not inferred from model size alone.

## Working hypothesis
The FPS drop is likely caused by one or more of these runtime factors:

1. NMS/detection-postprocess operations embedded in the TFLite graph are not fully accelerated by the GPU delegate.
2. Mixed GPU/CPU execution introduces synchronization overhead even when the convolutional backbone is smaller.
3. The 640 candidate also drives the app analysis resolution to the 960x720 path, increasing camera bitmap conversion and resize cost before inference.
4. File size is not a sufficient predictor of TFLite runtime speed when the operator graph changes.

## Current code support added for retest
`YoloDetector.setup()` now logs a single `VIA_GPU model_io` line after interpreter initialization. The line includes:

- model file name
- selected runtime backend
- requested GPU flag
- GPU delegate compatibility report
- input shape and dtype
- output shape and dtype
- normalized row/column interpretation
- parser layout name
- label count

Expected YOLO26n 7cls v2 640 log signature:

```text
layout=batched_nms_xyxy_score_class labels=7 output=[1, 300, 6]
```

A raw YOLO export should instead show a class-score layout such as `layout=class_score_cxcywh` with columns equal to `4 + classCount` after normalization.

## S25+ realtime retest — 2026-05-15
After adding `VIA_PERF` logging, the installed debug build was tested with the delivered YOLO26n exports. The camera was running live, GPU was used for float16 profiles, and int8 profiles used CPU. Results below use the stable samples after warmup.

| Model | Backend | Analysis | Processed FPS | Avg latency | Detect stage |
| --- | --- | --- | ---: | ---: | ---: |
| `best_yolo26n_7cls_v2_float16_640.tflite` | GPU | 960x720 | 4.87 | 204ms | ~204ms |
| `best_yolo26n_7cls_v2_float16_512.tflite` | GPU | 768x576 | 7.45 | 130ms | ~130ms |
| `best_yolo26n_7cls_v2_float16_448.tflite` | GPU | 640x480 | 9.61 | 99ms | ~99ms |
| `best_yolo26n_7cls_v2_float16_416.tflite` | GPU | 640x480 | 10.93 | 86ms | ~86ms |
| `best_yolo26n_7cls_v2_float16_320.tflite` | GPU | 480x360 | 16.61 | 51ms | ~51ms |
| `best_yolo26n_7cls_v2_int8_640.tflite` | CPU | 960x720 | 6.00 | 161ms | ~162ms |
| `best_yolo26n_7cls_v2_int8_320.tflite` | CPU | 480x360 | 20.07 | 38ms | ~38ms |

Conclusion: lowering the automatic startup profile to `int8_320` would only hide the regression and is not the intended fix. The default recommendation should remain pointed at the high-resolution YOLO26n candidate while this issue tracks why the delivered 640 export is slow. The low-resolution/int8 rows are useful only as diagnostic baselines showing that the rest of the camera pipeline can run near 20 FPS when the TFLite graph is cheap enough.

The evidence points back to the export contract: the YOLO26n 640 file is a NMS/top-k TFLite graph (`[1, 300, 6]`) rather than the older raw class-score graph (`[1, 11, 8400]`). Recovering high-resolution throughput likely requires a NMS-free/raw-output YOLO26n export, or another export/runtime change that removes the detection-postprocess bottleneck.


## Raw-output delivery received — 2026-05-15
A follow-up package `android_delivery_yolo26n_7cls_v2_raw_output.zip` was received and preserved under `resources/models/yolo26n_7cls_v2_raw_output/`. Its handoff explains that the previous YOLO26n TFLite files were exported with `nms=False` but still followed the YOLO26 `end2end=True` branch, producing `[1, 300, 6]` and postprocess-like operators.

The new package forces `end2end=False` and exports raw tensors. Local flatbuffer inspection confirms `best_yolo26n_7cls_v2_raw_float16_640.tflite` has `output=[1, 11, 8400]` and `TOPK_V2/GATHER_ND/TILE/FLOOR_MOD/GATHER = 0`, matching the previous YOLO11n 7cls raw-output contract.

## Raw-output realtime retest — 2026-05-15
The app was rebuilt with the raw-output assets and installed for a live-camera check. The 640 float16/GPU candidate now reports the raw parser path:

```text
model_io model=best_yolo26n_7cls_v2_raw_float16_640.tflite backend=GPU requestedGpu=true compat=true input=[1, 640, 640, 3] output=[1, 11, 8400] rows=8400 cols=11 transposed=false layout=class_score_cxcywh labels=7
```

Compared with the NMS-included 640 result above, the raw-output 640 result confirms the export-contract diagnosis:

| Model | Backend | Analysis | Output | Processed FPS | Detect stage |
| --- | --- | --- | --- | ---: | ---: |
| `best_yolo26n_7cls_v2_float16_640.tflite` | GPU | 960x720 | `[1, 300, 6]` | 4.87 | ~204ms |
| `best_yolo26n_7cls_v2_raw_float16_640.tflite` | GPU | 960x720 | `[1, 11, 8400]` | 20.06..21.24 in the first stable window | ~43ms |

After continued running on the same warmed device, thermal service reported `Thermal Status: 2`; the 640 raw path then fell to roughly 9..12 processed FPS with detect around 90ms. That later drop is a device/thermal-state observation, not evidence that the raw-output export still contains the previous NMS/TopK bottleneck.

Follow-up spot checks under the same warmed state:

| Model | Backend | Analysis | Processed FPS | Detect stage |
| --- | --- | --- | ---: | ---: |
| `best_yolo26n_7cls_v2_raw_float16_512.tflite` | GPU | 768x576 | 13.4..13.8 | ~59..62ms |
| `best_yolo26n_7cls_v2_raw_float16_416.tflite` | GPU | 640x480 | 17.0..17.5 | ~46..48ms |

Conclusion: Android delivery should use the raw-output export plus app-side confidence/NMS. Do not lower the automatic recommendation to 320/int8 merely to hide a high-resolution regression; low-resolution/int8 remains a diagnostic/manual fallback only. Sustained outdoor FPS should be interpreted together with thermal state.

## Input-rate metric correction — 2026-05-15
One measurement caveat from the raw-output retest was fixed before using the debug UI for further FPS decisions. The previous Camera FPS label was marked from the CameraX `ImageAnalysis` callback. Since that callback held the current `ImageProxy` open until processing copied and closed it, slow inference could backpressure the analyzer and make the input label fall with Processed FPS. Replay FPS had a similar issue because it was marked only when the processing loop captured a replay bitmap.

The input-rate label now uses source-frame events:

- live camera: Camera2 preview session capture callback
- replay: `TextureView.onSurfaceTextureUpdated`
- analyzer callback: fallback only until the first preview capture callback is observed

Post-fix live-camera evidence with raw 640/GPU while processing was thermally slow:

| Input label | Processed label | Interpretation |
| ---: | ---: | --- |
| Camera FPS 29.82 | Processed FPS 9.67 | Camera capture still near 30fps; inference path is saturated |
| Camera FPS 29.85 | Processed FPS 9.52 | Input metric no longer follows model latency |
| Camera FPS 29.98 | Processed FPS 9.57 | Debug UI now separates capture cadence from throughput |

## 30fps-first runtime policy — 2026-05-15
Further field observation showed raw 640 float16/GPU tends to process around the 20fps range, while raw 512 float16/GPU can reach the 30fps target. Because signal continuity is more important than maximum input resolution for this guidance path, startup recommendation and onboarding calibration now prioritize the highest candidate that can meet a 30fps target.

Applied policy:

- GPU float16 startup recommendation target input is 512.
- Legacy/saved GPU selections above the new frame-priority recommendation are replaced at startup.
- Onboarding calibration target is 30fps, so a 640 result around 20fps no longer passes.
- Replay input display is capped to the app target, `Replay FPS: 30.00`, even when the source video is 60fps; `Processed FPS` remains the model throughput metric.

## Acceptance criteria for closing #58
- The 640 FPS regression is attributed to either export layout/delegate compatibility, analysis resolution cost, or model compute cost with evidence.
- The team decides whether Android should use NMS-included or NMS-free TFLite exports.
- Default model recommendation is frame-priority based rather than lowered blindly: raw 512 is preferred because raw 640 was observed around 20fps while 512 can meet the 30fps target.

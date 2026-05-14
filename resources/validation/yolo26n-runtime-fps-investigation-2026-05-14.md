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

## Retest plan
1. Run the same realtime route with YOLO26n float16 320/416/448/512/640.
2. Record Camera FPS, Processed FPS, latency, model name, backend, analysis resolution, and the new `model_io` log line.
3. Compare YOLO26n NMS-included 640 against a future NMS-free YOLO26n 640 export if available.
4. Keep the 320 candidate as a temporary field-test workaround only; do not permanently lower the default until the export/runtime cause is decided.
5. If NMS-free YOLO26n restores expected throughput, prefer raw-output export plus app-side NMS for Android delivery.

## Acceptance criteria for closing #58
- The 640 FPS regression is attributed to either export layout/delegate compatibility, analysis resolution cost, or model compute cost with evidence.
- The team decides whether Android should use NMS-included or NMS-free TFLite exports.
- Default model recommendation is updated only after measured FPS and detection stability are compared across candidate resolutions.

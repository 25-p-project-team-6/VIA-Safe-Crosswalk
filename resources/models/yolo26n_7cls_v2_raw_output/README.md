# YOLO26n 7cls v2 raw-output delivery notes

This folder preserves the Android delivery metadata for the NMS-free / TopK-free
YOLO26n 7cls v2 export used to investigate Issue #58.

- `RAW_OUTPUT_ANDROID_HANDOFF.md`: source handoff and export rationale
- `manifest.txt`: exported TFLite names, sizes, input shapes, raw output shapes, and postprocess-op counts
- `SHA256SUMS.txt`: original delivery-package checksums; paths refer to the source package layout
- `app-assets.SHA256SUMS.txt`: checksums for the raw-output TFLite files copied into `app/src/main/assets/`

The actual app-bundled TFLite files live under `app/src/main/assets/`.

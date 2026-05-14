# YOLO26n 7cls v2 delivery notes

This folder preserves the Android delivery metadata for Issue #54.

- `ANDROID_APP_AGENT_HANDOFF.md`: source package handoff and adoption rationale
- `comparison_yolo26n_vs_yolo11n_7cls_v2.md`: metric comparison used for adoption
- `manifest.txt`: exported TFLite names, sizes, input shapes, and output shape
- `SHA256SUMS.txt`: original delivery-package checksums; paths refer to the source package layout
- `app-assets.SHA256SUMS.txt`: checksums for the TFLite files copied into `app/src/main/assets/`

The actual app-bundled TFLite files live under `app/src/main/assets/`.

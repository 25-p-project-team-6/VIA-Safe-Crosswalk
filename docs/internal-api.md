# Internal API Notes

This document summarizes the app-internal APIs that are useful for maintainers and tests. It is not a stable external SDK contract.

## Frame processing boundary

`MainActivity` owns the production runtime loop:

1. receive the latest `ImageProxy` from CameraX,
2. close stale frames and process only the newest pending frame,
3. convert/rotate to `Bitmap`,
4. call `YoloDetector.detect`,
5. collect `CrossingSupportManager.snapshot`,
6. call `SignalAnalyzer.analyze`,
7. stabilize guidance and update UI/feedback.

Keep expensive work off the analyzer callback path. Camera/input FPS and processed FPS are intentionally separate measurements.

## ML and guidance components

- `YoloDetector`: model lifecycle and LiteRT inference. Prefer constructing it at activity/model-selection boundaries, not per frame.
- `YoloOutputParser`: raw tensor parser for app-side confidence/NMS. Update parser tests when adding a new export layout.
- `DetectionLabels`: central label schema and model-file filtering. Keep class order aligned with model delivery manifests.
- `PostProcessor`: color-corrects traffic-light detections and bridges into `TrafficLightStateTracker`.
- `TrafficLightStateTracker`: time-based stable state machine for red/green/unknown.
- `ObjectTracker`: target-signal choice and continuity scoring.
- `SignalAnalyzer`: composition root for target selection, state tracking, walk policy, occupancy caution, advisory flags, and debug output.
- `ConservativeWalkSignalPolicy`: user guidance phase machine. Motion/location context is support-only and must not create `GO` from `UNKNOWN` or startup green.
- `GuidanceStateStabilizer`: final user-visible debounce layer.

## Context components

- `CrossingSupportManager` registers gyroscope, gravity, and location listeners and emits immutable `CrossingSupportSnapshot` values.
- `CrossingSupportSnapshot.supportsWalkContinuation` is continuity evidence only.
- `MapProximityManager` combines bundled tiles, optional refreshed tiles, and live OSM fetches before asking `MapProximityEngine` for proximity state.
- `MapTileStore` and `OsmNearbyCrossingFetcher` touch local/remote map data; keep tests deterministic by isolating fetch/cache behavior.

## Feedback and UX components

- `SignalFeedbackPolicy` decides when a state change or repeat interval should emit feedback.
- `SignalFeedbackManager` owns Android TTS/vibration integration.
- `UsageGuideContent` and `PracticeFeedbackPlayer` mirror safety copy and feedback practice examples.
- `OnboardingPermissionPolicy` defines required permissions; keep copy concise because this is the first-run accessibility path.

## Common test seams

Prefer unit-testing policy and parser components without Android framework dependencies:

- traffic state timing: `TrafficLightStateTrackerTest`,
- walk guidance phase behavior: `ConservativeWalkSignalPolicyTest`,
- output tensor parsing: `YoloOutputParserTest`,
- context snapshot derivations: `CrossingSupportSnapshotTest`,
- feedback repeat timing: `SignalFeedbackPolicyTest`,
- map/geometric proximity: `MapProximityEngineTest`, `MapTileStoreParsingTest`.

## Change checklist

- If changing a threshold, update or add a focused test and note the reason in code or validation docs.
- If changing model filenames or class order, update `DetectionLabels`, model resources metadata, parser tests, and onboarding calibration expectations.
- If changing network/location behavior, update permission/privacy notes and add deterministic cache/fetch tests where possible.
- If changing user guidance states, update policy tests, stabilizer tests, feedback tests, and usage-guide copy together.

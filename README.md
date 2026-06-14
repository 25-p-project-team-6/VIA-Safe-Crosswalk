# VIA Safe Crosswalk

VIA Safe Crosswalk is an Android prototype that helps blind and low-vision pedestrians reason about signalized crosswalks. It combines live CameraX frames, on-device TensorFlow Lite/LiteRT object detection, conservative walk-signal policy logic, map/GPS proximity context, and Korean voice/haptic feedback.

> Safety boundary: this app is an assistive prototype, not a certified mobility aid or a replacement for traffic laws, cane/dog guidance, audible pedestrian signals, or human judgment. Visual traffic-signal recognition can fail because of occlusion, lighting, model error, GPS drift, thermal throttling, or camera framing.

## Current implementation scope

- CameraX live preview and analysis pipeline with separate input and processed FPS tracking.
- On-device YOLO-family TFLite inference through Google AI Edge LiteRT, with GPU preference for float models and CPU fallback.
- App-side YOLO raw-output parsing, confidence filtering, NMS, and detection label normalization.
- HSV-assisted pedestrian-signal color correction and time-based traffic-light state stability.
- Conservative walk guidance that requires a red baseline before announcing a green transition.
- Risk/crosswalk-occupancy blocking so central vehicles or occupied crosswalk cues can suppress or downgrade `GO`.
- Gyroscope, gravity/tilt, GPS, and map proximity context used only as supporting continuity evidence.
- Korean TTS/vibration feedback for `WAIT`, `STOP`, `GO`, and caution/advisory states.
- Onboarding, model calibration/selection, permission gating, usage guide, emergency-contact SMS flow, and debug map/tools.

## Repository layout

```text
.
├── app/                         # Android application module
│   └── src/main/java/kr/co/gachon/pproject6/via/
│       ├── MainActivity.kt       # Runtime orchestration: camera -> detector -> analyzer -> UI/feedback
│       ├── camera/               # CameraX and camera-rate policies
│       ├── context/              # crossing support, map proximity, bundled map tiles
│       ├── feedback/             # TTS/haptic timing and playback
│       ├── guide/                # usage guide and practice feedback
│       ├── input/                # remote button press classification
│       ├── map/                  # debug map, OSM fetch/cache helpers
│       ├── ml/                   # detector parser, state tracking, policy, risk/advisory logic
│       ├── onboarding/           # permissions and calibration flow
│       ├── safety/               # emergency contact/message support
│       ├── settings/             # settings screens
│       ├── ui/                   # overlay rendering
│       └── util/                 # image, performance, rate, phone utilities
├── gradle/libs.versions.toml     # Version catalog
├── resources/                    # Model, validation, and decision records used by the app repo
└── README.md                     # This file
```

Additional project-wide handoff/spec documents may exist one directory above this app repo under `../resources/decisions/` when working from the larger workspace.

## Requirements

- Android Studio with Android Gradle Plugin 8.9.1 support.
- JDK 17 is recommended for AGP 8.x projects.
- Android SDK platforms/tools for `compileSdk = 35` and `targetSdk = 35`.
- A device or emulator running Android 7.0+ (`minSdk = 24`). A physical device is required for useful camera, vibration, GPS, and sensor validation.
- Windows users can run the checked-in `gradlew.bat`; Linux/macOS users can run `./gradlew`.

## Build, test, and install

From the `pproject/` directory:

```bash
# Unit tests
./gradlew testDebugUnitTest

# Android lint
./gradlew lintDebug

# Debug APK
./gradlew assembleDebug

# Optional device install
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

On Windows PowerShell or CMD, replace `./gradlew` with `./gradlew.bat`:

```bat
gradlew.bat clean
gradlew.bat testDebugUnitTest
gradlew.bat lintDebug
gradlew.bat assembleDebug
adb install -r app\build\outputs\apk\debug\app-debug.apk
```

Build outputs are expected under `app/build/...`. Do not run multiple Gradle invocations in parallel against the same checkout/build directory.

## Major dependencies

Versions are declared in `gradle/libs.versions.toml`.

| Area | Libraries / platform APIs |
| --- | --- |
| Android app stack | AndroidX Core KTX, AppCompat, Activity, ConstraintLayout, Material Components |
| Camera | CameraX core, camera2, lifecycle, view |
| ML inference | Google AI Edge LiteRT, LiteRT GPU, LiteRT GPU API |
| Sensors/location | Android gyroscope, gravity sensor, `LocationManager` GPS/network providers |
| Tests | JUnit 4, AndroidX test JUnit, Espresso |

The app also uses Android platform TTS, vibration, SMS intent/permission APIs, and a small bundled Leaflet asset for debug map UI.

## Runtime architecture

```text
CameraX ImageAnalysis
  -> MainActivity latest-frame drain loop
  -> ImageUtils bitmap conversion/rotation
  -> YoloDetector + YoloOutputParser
  -> SignalAnalyzer
       -> PostProcessor HSV color correction
       -> ObjectTracker target selection
       -> TrafficLightStateTracker stable red/green/unknown
       -> ConservativeWalkSignalPolicy walk guidance state machine
       -> Risk/occupancy/advisory evaluators
       -> CrossingSupportSnapshot from sensors/GPS/map context
  -> GuidanceStateStabilizer
  -> OverlayView + debug panel + SignalFeedbackManager
```

### Guidance policy in brief

- Startup green is treated as `WAIT` until a stable red baseline has been seen.
- Stable red moves the policy to a ready state and emits `STOP`.
- Red-to-green transition can emit `GO` only if no risk/caution policy blocks it.
- Unknown signal periods during an active crossing use grace windows instead of immediately resetting the session.
- Gyro/GPS/tilt/map context may extend continuity or support handoff decisions, but it must not create `GO` by itself.
- A stable red or explicit reset evidence returns the policy to a safer non-go state.

## Internal API surface

These Kotlin classes are the main extension and test seams. They are internal app APIs, not a separately versioned SDK.

| API | Responsibility | Typical caller |
| --- | --- | --- |
| `ml.YoloDetector` | Load a TFLite asset, choose delegate, run inference, return overlay boxes. | `MainActivity` |
| `ml.YoloOutputParser` | Parse raw `[1, 11, anchors]` YOLO tensors into normalized boxes. | `YoloDetector` |
| `ml.InferenceModelProfile` | Infer quantization/input size and recommended analysis resolution from model filenames. | onboarding/model selection |
| `ml.SignalAnalyzer` | Convert raw detections and context into signal analysis, guidance, target, risk, and debug data. | `MainActivity` |
| `ml.TrafficLightStateTracker` | Time-stabilize red/green/unknown signal states. | `PostProcessor` |
| `ml.ConservativeWalkSignalPolicy` | Maintain `WAITING_FOR_RED_BASELINE`, `READY_FOR_GREEN_TRANSITION`, and `WALK_ALLOWED` phases. | `SignalAnalyzer` |
| `ml.GuidanceStateStabilizer` | Debounce user-visible guidance changes. | `MainActivity` |
| `context.CrossingSupportManager` | Register sensors/location and expose a `CrossingSupportSnapshot`. | `MainActivity` |
| `context.MapProximityManager` / `MapProximityEngine` | Merge bundled/remote crosswalk features and compute proximity/handoff context. | `CrossingSupportManager`, debug map |
| `feedback.SignalFeedbackManager` / `SignalFeedbackPolicy` | Rate-limit and play TTS/haptic feedback. | `MainActivity`, usage guide |
| `onboarding.OnboardingPermissionPolicy` / `CalibrationSelector` | Gate permissions and select calibrated model candidates. | onboarding flow |

### Example: analyzing one frame

```kotlin
val boxes = detector.detect(rotatedBitmap)
val support = crossingSupportManager.snapshot()
val raw = signalAnalyzer.analyze(
    bitmap = rotatedBitmap,
    rawBoxes = boxes,
    enableTrafficLogic = true,
    enableHighlight = true,
    crossingSupportSnapshot = support
)
val stable = guidanceStateStabilizer.update(raw.userGuidanceState)
```

### Example: policy-only unit testing

```kotlin
val policy = ConservativeWalkSignalPolicy(GuidanceTuningDefaults.walkSignalConfig)
policy.update(TrafficLightState.RED, CrossingSupportSnapshot())
val decision = policy.update(TrafficLightState.GREEN, CrossingSupportSnapshot())
check(decision.state == UserGuidanceState.GO)
```

Most policy, parser, context, and feedback classes have focused unit tests under `app/src/test/java/kr/co/gachon/pproject6/via/`.

## Permissions and privacy-relevant behavior

Declared permissions include:

- `CAMERA`: required for real-time signal detection.
- `ACCESS_FINE_LOCATION` / `ACCESS_COARSE_LOCATION`: used for crossing continuity and map proximity support.
- `VIBRATE`: used for haptic guidance.
- `SEND_SMS`: used by the emergency-contact flow.
- `INTERNET`: used for live OSM/Overpass-style nearby crossing lookups and configured map APIs.

The core detector runs on-device against bundled TFLite assets. Location-derived map support can use bundled map tiles and may also perform runtime network fetches for nearby OSM crossings; review `context/MapProximityManager.kt`, `context/MapTileStore.kt`, and `map/OsmNearbyCrossingFetcher.kt` before changing privacy or networking behavior.

## Models and provenance

Bundled TFLite files live in `app/src/main/assets/`. Delivery metadata, manifests, and checksums are preserved under:

- `resources/models/yolo26n_7cls_v2/`
- `resources/models/yolo26n_7cls_v2_raw_output/`
- `resources/validation/`

The raw-output YOLO26n files are exported without built-in NMS/TopK so the Android app can use the same app-side parser/NMS path as earlier raw YOLO exports. See `resources/models/yolo26n_7cls_v2_raw_output/RAW_OUTPUT_ANDROID_HANDOFF.md` for class order, tensor shapes, and export rationale.

## Safety and validation notes

Before claiming runtime behavior changes complete, prefer:

```bash
./gradlew clean
./gradlew testDebugUnitTest
./gradlew lintDebug
./gradlew assembleDebug
```

Then validate on a physical device outdoors or with representative replay footage. Pay special attention to:

- false `GO` prevention when the app starts on green,
- red/green flicker and LED/camera artifacts,
- brief signal loss while already crossing,
- GPS/map drift near adjacent crosswalks,
- thermal throttling effects on processed FPS,
- risk-object and occupancy blocking behavior,
- TTS/haptic repeat timing and accessibility clarity.

## License

No repository-level open-source license file is currently present in this checkout. Treat the code, bundled models, map data, and documentation as all-rights-reserved/internal project material until the project owner adds an explicit license and third-party attribution policy.

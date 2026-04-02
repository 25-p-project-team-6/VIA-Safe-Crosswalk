# PR Notes

## Suggested title
Stabilize map-backed crosswalk matching and debug map behavior

## Summary
- prewarm the kinetic map guest session from app startup instead of waiting for the debug map screen
- merge bundled crosswalk data and nearby OSM crossings into one live matching stream
- deduplicate overlapping bundled/OSM crossings and expose `bundled` / `osm` / `hybrid` match sources in debug output
- update map matching so the nearest in-range candidate can take over after consistent fixes instead of holding stale matches too long
- keep the debug map zoom level during live location updates and add a recenter button
- cache map tiles and OSM nearby results, plus add a debug action to clear map caches
- reduce first-launch camera stalls after onboarding calibration by releasing onboarding camera resources and retrying cold-start camera bind once

## Testing
- `./gradlew.bat :app:testDebugUnitTest --tests "kr.co.gachon.pproject6.via.context.MapProximityEngineTest"`
- `./gradlew.bat :app:testDebugUnitTest --tests "kr.co.gachon.pproject6.via.context.CrossingSupportSnapshotTest"`
- `./gradlew.bat testDebugUnitTest lintDebug assembleDebug`
- manual APK install on connected Android device via `adb install --no-streaming -r`

## Not tested enough
- long outdoor walks with dense adjacent crossings
- prolonged OSM cache expiration / recovery behavior
- repeated onboarding recalibration on multiple devices

## Main commits in this stack
- `7130675` Keep map debugging stable during live movement and recover first-launch camera stalls
- `1de800b` Keep nearby crosswalk overlays visible and make map caches resettable
- `a4adb51` Make map matching update continuously instead of waiting for the debug screen
- `4aeea09` Prefer the nearest bundled crosswalk as soon as it becomes the best live candidate
- `d083f7f` Merge bundled and OSM crossings into one live match stream
- `295bb71` Warm map sessions from process start and stop dropping OSM context while walking

## Reviewer focus
- whether bundled/OSM overlap merge distance is appropriate
- whether nearest-candidate switching is stable enough for real walking GPS noise
- whether app-start session prewarm has any unwanted startup cost

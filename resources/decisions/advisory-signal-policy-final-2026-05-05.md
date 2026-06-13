# Advisory signal policy finalization — 2026-05-05

## Intent
VIA is an assistive guidance tool. It describes the signal and crosswalk context it currently observes, but it must not sound like it is making the final crossing decision for the user.

This document closes issue #26 by recording the final advisory wording, source priority, conservative transition rules, ambiguity handling, vehicle policy, and quantitative evaluation plan that the app implementation should preserve.

## Runtime source priority
1. **Vision / pedestrian-signal detection is primary truth.** `human_green` and `human_red` detections are the only inputs allowed to confirm a pedestrian signal color.
2. **Matched crosswalk cluster and GPS are support context.** They can explain whether the same crossing context is stable or whether a fresh signal cycle should be requested, but they cannot create a green confirmation by themselves.
3. **Tilt, gyro, and movement are continuity support only.** They extend short visual-loss handling while walking, especially looking-down cases, but they do not override the visual signal state.
4. **Bluetooth button, nearby-crosswalk guidance, and emergency SMS are separate assistive actions.** They must not be interpreted as a signal permission state.

## User-facing advisory states
| Internal state | Screen title | Speech text | Meaning |
| --- | --- | --- | --- |
| `RED_CONFIRMED` | 보행자 신호 빨간색으로 보임 | 빨간불이 확인됩니다 | A stable pedestrian red signal is visible. |
| `GREEN_CONFIRMED` | 보행자 신호 초록으로 보임 | 초록불이 확인됩니다 | A stable pedestrian green signal is visible and ambiguity gates passed. |
| `GREEN_WITH_CAUTION` | 초록으로 보이나 주의 필요 | 초록불이 확인되지만 차량 점유 가능성이 있습니다 | Green is visible, but crosswalk occupancy suggests caution. |
| `TRANSITION_WAIT` | 다음 신호 대기 권장 | 다음 신호 전환을 기다립니다 | Current green is not yet tied to a safe fresh cycle for this crossing context. |
| `UNCERTAIN_VIEW` | 신호 확인 불확실 | 신호 확인이 불안정합니다 | The app needs a clearer pedestrian signal view or lower ambiguity before confirming. |

The wording intentionally avoids command phrases such as "건너세요", "멈추세요", or "건너도 됩니다".

## Confidence model
Confidence is displayed as `신뢰 높음 / 신뢰 보통 / 신뢰 낮음` and logged with a numeric score.

- High: score >= 75
- Medium: score >= 55
- Low: score < 55

Positive evidence includes stable pedestrian signal state, target score, and stable matched cluster context. Penalties are applied for multiple pedestrian signals, visible vehicle signals, small target/zoom need, recent target reacquire, cluster changes, missing map match, signal-loss grace, and occupancy caution.

## Conservative green policy
A green confirmation must satisfy all of these runtime gates:

- Stable pedestrian green state exists.
- Red baseline / fresh cycle policy is satisfied.
- Confidence is not low.
- Multiple pedestrian signals are not present.
- Vehicle traffic-light detections are not present in the current view.
- Target is not too small and does not require zoom guidance.
- Target was not just reacquired.
- Matched cluster did not recently churn.

If any gate fails, the app falls back to `TRANSITION_WAIT` or `UNCERTAIN_VIEW` with a reason instead of confirming green.

## Cluster and movement policy
- Moving far enough to a new crossing context resets the policy toward a fresh red baseline before a later green can be confirmed.
- A same-crossing visual-loss gap may keep continuity for a short configured grace window.
- GPS, matched cluster, gyro, and tilt can extend or explain continuity; they cannot independently confirm green.
- Cluster churn is treated as ambiguity because the app may be looking at a different crosswalk or signal group.

## Ambiguity feedback
The app should explain common uncertainty causes in user-facing or debug output:

- Multiple pedestrian signals: ask the user to center one signal.
- Vehicle-only or vehicle-signal-visible view: ask for a pedestrian signal.
- Small/occluded signal: ask the user to move closer or zoom.
- Recent reacquire: avoid immediately promoting the new view to confirmed green.
- Matched cluster missing or changed: surface that the crosswalk reference is weak or has changed.
- Signal lost during a previous green: explain that the green signal is being rechecked.

## Vehicle policy
Vehicle-signal detections are not pedestrian-signal truth. A vehicle-only view stays uncertain and asks for a pedestrian signal. During green, visible vehicle signals block `GREEN_CONFIRMED`. During red, a stable pedestrian red can remain `RED_CONFIRMED`, but the vehicle-signal reason still reduces confidence for evaluation/debugging. Crosswalk occupancy affects green as `GREEN_WITH_CAUTION`; it does not turn a red phase into a green/walk state.

## Quantitative evaluation plan
Evaluation should run in parallel across offline replay, labeling, and field trial logs.

### Required labels
For each sampled frame or event window, label:

- pedestrian signal truth: `human_red`, `human_green`, `unknown`, or `not_visible`
- vehicle signal truth: visible/not visible and color when applicable
- number of visible pedestrian signal candidates
- target box quality: centered, small, occluded, reacquired
- matched cluster state: stable, missing, changed, new crossing, same crossing
- occupancy: vehicle/person/bicycle/motorcycle in crosswalk area
- expected advisory state and expected confidence band

### Model metrics
- Human red / human green / unknown classification accuracy.
- Precision and recall for `human_red` and `human_green`.
- Vehicle-signal-to-pedestrian-signal confusion rate.
- Small or occluded signal uncertainty rate.
- Multiple-signal false confirmation rate.

### Policy metrics
- False `GREEN_CONFIRMED` rate when ground truth is red/unknown/vehicle-only.
- Missed `GREEN_CONFIRMED` rate when ground truth is clear pedestrian green and no ambiguity gates apply.
- `TRANSITION_WAIT` correctness after red-baseline, cluster-change, and handoff events.
- Cluster transition error rate: new crossing treated as same crossing, or same crossing treated as new crossing.
- Signal-loss grace recovery correctness.
- `GREEN_WITH_CAUTION` precision/recall for crosswalk occupancy.

### Field trial checklist
For each route, save the replay video, app debug logs, GPS/map context, selected model profile, and observer notes. Review at least: straight crosswalk, adjacent multiple signals, vehicle signal near pedestrian signal, occluded/small signal, cluster transition after walking, and looking-down visual-loss recovery.

## Release acceptance gates
- No user-facing advisory string uses command-style crossing instructions.
- Every advisory log includes state, confidence band/score, ambiguity reasons, pedestrian/vehicle signal counts, zoom/reacquire flags, and cluster-change count.
- Offline replay reports the model and policy metrics above.
- Field-trial notes document known failure cases and remaining risk.
- Onboarding and usage guide keep the assistive-tool disclaimer visible.

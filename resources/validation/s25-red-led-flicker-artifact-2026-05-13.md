# S25+ red LED flicker artifact 검증 지침 — 2026-05-13

## 관련 이슈
- GitHub Issue: #53
- 대상 현상: S25+ 카메라 영상에서 실제 빨간 보행자 신호가 켜져 있는데도 일부 프레임에서 빨간 LED가 사라져 꺼진 신호처럼 보이는 현상

## 판정
이 현상은 우선 모델 학습 오류가 아니라 **촬영 단계 camera artifact**로 분류한다.

가능성이 높은 원인은 다음 조합이다.

1. LED 신호등의 PWM/멀티플렉싱 점멸
2. 주간 자동 노출의 짧은 셔터 속도
3. 60fps 샘플링과 LED 점멸 주기의 위상 충돌
4. CMOS 롤링셔터 readout
5. HDR/HLG, temporal denoise, EIS, 자동 FPS 같은 스마트폰 후처리
6. HEVC/H.265 압축에서 작은 고채도 빨간 영역이 손실되는 현상

## 라벨링 지침
- 실제 상태가 red인데 카메라 프레임에서만 빨간 LED가 빠진 구간은 `off`로 라벨링하지 않는다.
- 전후 프레임에서 red가 유지되고 green이 명확히 등장하지 않았다면 `camera-induced missing red`, `transient unknown`, 또는 `flicker artifact`로 별도 기록한다.
- 단일 프레임만 보고 red/off를 결정하지 않는다.
- 학습/검증 표에는 artifact 빈도와 촬영 조건(fps, HDR/HLG, 자동 FPS, exposure/셔터, codec)을 함께 기록한다.

## 앱 런타임 대응
현재 앱은 단일 프레임 판단 대신 시간축 안정화를 사용한다.

- `TrafficLightStateTracker`는 확정된 red를 `5,000ms` 동안 UNKNOWN에서 유지한다.
- #53 대응으로 확정 전 red 후보도 `150ms` 이하의 짧은 UNKNOWN gap을 bridge한다.
- 이 bridge는 **red 후보에만** 적용한다. Sparse green evidence가 GO 허가로 이어질 수 있으므로 green 후보는 UNKNOWN gap에서 계속 리셋한다.
- red가 일시적으로 missing되어도 green이 지속 확인되지 않으면 off/GO로 바꾸지 않고 red 또는 unknown으로 보수 처리한다.

## 현장 QA 체크리스트
아래 항목을 S25+ 영상별로 기록한다.

| 영상/장소 | fps | HDR/HLG | codec | red missing 연속 길이 | 전후 red 유지 | green 등장 여부 | 앱 stable state | 판단 |
| --- | ---: | --- | --- | ---: | --- | --- | --- | --- |
|  | 60/30 | on/off | HEVC/H.264 |  |  |  | RED/UNKNOWN |  |

검증 시 우선순위:
1. 원본 입력 영상에서 red가 실제로 프레임 단위로 missing되는지 확인한다.
2. missing 전후 red가 유지되는지 확인한다.
3. HDR/HLG/자동 FPS/codec 설정을 바꿔 재현 빈도가 줄어드는지 비교한다.
4. 앱의 stable `trafficState`가 short missing frame에서 RED를 유지하거나 UNKNOWN으로 보수 처리하는지 확인한다.

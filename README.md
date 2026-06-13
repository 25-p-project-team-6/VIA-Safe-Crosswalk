# VIA Safe Crosswalk

시각장애인의 안전한 보행을 돕기 위한 Android 기반 실시간 신호등 인식 프로젝트입니다.

## 현재 구현 범위
- CameraX 기반 실시간 카메라 입력
- TensorFlow Lite 기반 온디바이스 추론
- 다중 TFLite 모델 선택 및 GPU 전환
- 신호등 색상 HSV 보정 후처리
- 타겟 신호등 추적 및 안정화된 상태 판정
- 보수적 보행 정책(초기 초록 무시, 적색→초록 전환 시에만 안내)
- 위험 객체 감지 시 `건너세요` 차단
- 상태별 시각 피드백(테두리 색상)
- 상태별 음성/TTS 및 진동 피드백
- 디버그 패널(FPS, latency, threshold, zoom)
- 현재 실사용 기본 모델은 `best_float16_448.tflite` (S25+ 측정상 640≈5fps, 448≈15fps)

## 아직 남은 주요 작업
- OCR 기반 잔여 시간 인식
- 차량/위험 요소 탐지 고도화 및 threshold 실기기 보정
- 위험도 산출 및 경고 정책 정교화
- 사용자용 접근성 UX 정리
- 최종 배포 직전 불필요 에셋 정리(현재 `.onnx`, `.pt`, `.ipynb`는 유지)

## 코드 구조
- `MainActivity.kt`: 앱 진입점, 카메라/모델/UI orchestration
- `camera/CameraManager.kt`: CameraX 관리
- `ml/YoloDetector.kt`: TFLite 추론
- `ml/SignalAnalyzer.kt`: 후처리/타겟 선택/신호 상태 분석
- `ml/PostProcessor.kt`: HSV 기반 색상 보정
- `ml/TrafficLightStateTracker.kt`: 신호 상태 안정화
- `feedback/SignalFeedbackManager.kt`: TTS/진동 피드백
- `ui/OverlayView.kt`: 바운딩 박스 렌더링
- `util/PerformanceTracker.kt`: 성능 통계

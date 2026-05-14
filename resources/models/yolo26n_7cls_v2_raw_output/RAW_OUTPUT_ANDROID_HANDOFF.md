# YOLO26n 7cls v2 RAW-output Android 전달 패키지

이 패키지는 기존 YOLO11n 앱 후처리 경로와 같은 조건으로 FPS 비교하기 위한 **NMS-free / TopK-free / raw-output** YOLO26n TFLite 세트입니다.

## 왜 새로 export했나

기존 전달 YOLO26n TFLite는 `nms=False`였지만 YOLO26 모델의 `end2end=True` branch가 유지되어 output이 `[1, 300, 6]` 형태였고, TFLite graph 내부에 `TOPK_V2`, `GATHER_ND`, `TILE`, `FLOOR_MOD` 등 postprocess 계열 op가 포함되었습니다.

Android 앱은 기존 YOLO11n에서 `[1, 11, 8400]` raw tensor를 받아 앱 내부에서 confidence/NMS를 수행하고 있었으므로, 동일 조건 비교를 위해 `end2end=False`를 강제해 raw tensor로 재-export했습니다.

## Export 조건

```text
format=tflite
nms=False
end2end=False
optimize=False
```

## 클래스 순서

```text
0 bicycle
1 motorcycle
2 vehicle
3 human_green
4 human_red
5 vehicle_green
6 vehicle_red
```

## 출력 형식

- 640 모델: `[1, 11, 8400]`
- 512 모델: `[1, 11, 5376]`
- 448 모델: `[1, 11, 4116]`
- 416 모델: `[1, 11, 3549]`
- 320 모델: `[1, 11, 2100]`

`11 = 4 bbox + 7 class scores` 입니다.

기존 YOLO11n raw parser가 `[1, 11, anchors]` 형태를 처리하고 있다면 같은 방식으로 연결하면 됩니다.

## 검증 포인트

모든 raw-output 파일에서 아래 postprocess op가 0개임을 확인했습니다.

```text
TOPK_V2=0
GATHER_ND=0
TILE=0
FLOOR_MOD=0
GATHER=0
```

## 우선 테스트 권장

1. `best_yolo26n_7cls_v2_raw_float16_640.tflite`
   - 기존 YOLO11n 640과 가장 같은 조건의 정확도/FPS 비교
2. `best_yolo26n_7cls_v2_raw_int8_640.tflite`
   - int8 640 속도 비교
3. `best_yolo26n_7cls_v2_raw_int8_320.tflite`
   - 저사양/최고 FPS 후보

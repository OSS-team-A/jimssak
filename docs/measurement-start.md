# 일반 휴대폰으로 크기 추정 시작하기

## 첫 목표와 범위

한 프레임에서 정면에 가까운 물체의 폭/높이(cm)를 계산하고 실측과 비교합니다.
첫 단계는 줄자로 잰 거리, 이후 단계는 모델이 예측한 거리를 넣어 각각 오차를 봅니다.
현재 실험 도구에는 학습/모델 추론/영상 추출/카메라 보정 기능이 없습니다.
두께, 부피, 무게는 현재 결과에서 구할 수 없습니다.

## 진행 순서

| 단계 | 할 일 | 완료 조건 |
|---|---|---|
| 1: 오늘 | 합성 JSON으로 계산/오차 보고서 실행 | 숫자와 단위, 보정 누락 처리 확인 |
| 2: 첫 실촬영 | 카메라 보정, 물체 3개 실측/촬영, 수동 bbox | 실제 데이터로 폭/높이 오차 기록 |
| 3: 깊이 모델 | 실내 metric depth 후보 한 개 연결 | 수동 거리와 모델 거리 오차 분리 비교 |
| 4: 통합 | 탐지 담당의 bbox/mask와 같은 프레임 사용 | 여러 거리/각도에서 재현 가능한 결과 |

첫 실험용 물체는 상자, 작은 수납장, 책장 등 윤곽이 분명한 물체 3개로 고릅니다.
의자나 투명/반사/가림이 큰 물체는 이후에 추가합니다.
잠정 목표는 정면 촬영 폭/높이 상대오차 20% 이내이며, 이는 달성 결과가 아닙니다.
전체 평균 외에 물체별 오차와 측정 실패 비율도 함께 보고합니다.

## 지금 실행

저장소 루트에서 기존 .venv 활성화 후:

```bash
python -m ai.measurement.experiment docs/examples/measurement-input-synthetic.json --truth docs/examples/measurement-truth-synthetic.csv
```

합성 예시는 fx=fy=1000px, 거리=2m, bbox 폭=512px/높이=504px입니다.
출력 폭=102.4cm, 높이=100.8cm. 가상 정답 100cm/100cm에 대해 MAE=1.6cm,
MAPE=1.6%입니다. 이 수치는 계산 검사용이며 실제 촬영 정확도가 아닙니다.

```bash
python -m ai.measurement.experiment docs/examples/measurement-input-synthetic.json --truth docs/examples/measurement-truth-synthetic.csv --output outputs/measurement/report-01.json
```

출력 경로는 새 파일이어야 합니다. 원본/기존 결과를 덮어쓰지 않습니다.
실제 촬영 결과와 위 합성 데이터를 구분해 보관하세요.

## 첫 촬영 준비

1. 같은 휴대폰의 후면 기본 카메라, 같은 해상도/줌을 유지합니다.
   초광각/디지털 줌/렌즈 자동 전환을 피하고, 보정 때와 같은 촬영 조건을 씁니다.
   영상 흔들림 보정의 동적 크롭은 내부 파라미터를 바꿀 수 있어 기록해야 합니다.
2. 폭과 높이를 줄자로 재고 cm로 기록합니다. 정답은 거리와 각도가 달라도 동일합니다.
3. 정면에서 가까운/중간/먼 거리로 각 한 장씩 촬영합니다. 물체 전체가 보여야 합니다.
4. 휴대폰 카메라 부근에서 물체 정면까지 거리를 m로 기록합니다.
   계산의 depth는 카메라 광축 방향 Z이므로 물체는 화면 중앙에 두고 정면 촬영합니다.
   첫 실험에서는 이 거리도 근사값입니다.
5. 체크보드 보정으로 원본 해상도 기준 fx/fy와 왜곡 계수를 구합니다.
   다양한 위치/기울기로 촬영하고 보정 오차를 확인합니다.
   픽셀 초점거리 fx/fy는 카메라 사양의 mm나 35mm 환산 초점거리와 다릅니다.
6. 렌즈 왜곡이 있으면 보정된 이미지의 bbox와 그 이미지의 새 카메라 행렬을 함께 사용합니다.
   원본 bbox와 보정 후 행렬을 섞지 않습니다. 현재 Frame에는 왜곡 계수가 없으므로
   이 전처리는 실험 입력 전에 해야 합니다.

## 실제 입력 만들기

합성 JSON을 `outputs/measurement/input-01.json`으로 복사해 실제 측정값으로 바꿉니다.
`mode`는 데이터 출처가 실제 모델일 때 model로 바꿉니다. 수동 bbox 실험은 demo로
유지하고 별도 실험 기록에 manual_bbox와 수동 거리 사용을 명시합니다.

- frame: 영상/프레임 ID, 원본 해상도, 보정된 fx_px/fy_px
- detection: 원본 좌표 bbox(x1,y1,x2,y2), track_id, 수동 거리 depth_m
- 정답 CSV: 같은 track_id, 줄자로 잰 width_cm/height_cm
- 첫 단계 bbox는 직접 지정해 탐지 오차를 분리합니다. confidence는 수동 설정이라고 기록합니다.
- 보정 또는 거리가 없으면 null로 둡니다. 임의 값을 실제 측정값으로 사용하지 않습니다.

CSV는 입력 JSON의 단일 프레임에 대응합니다. track_id를 중복하지 않습니다.
평가에 지정한 물품만 비교하며, 보정 누락 물품은 unavailable_track_ids로 표시됩니다.
MAE/MAPE는 계산 가능한 폭/높이에 대한 평균입니다. 실패가 제외되므로 측정 개수도 봅니다.
여러 사진은 파일별로 평가하고 파일명/조건을 실험 기록에 함께 남깁니다.

## 다음에 깊이 모델 붙이기

기본 Depth Anything V2는 상대 깊이이며, 이를 그대로 depth_m에 넣으면 안 됩니다.
실내용 metric 모델을 후보로 검증하고 실제 거리와 비교합니다.
마스크 내부 유효 깊이의 중앙값 등을 검토하되 가림/배경 혼입/경사 물체의 오차를 따로 봅니다.
미터 단위 출력도 실제 거리 정확도를 보장하지 않습니다.

너의 담당 입력: frame, camera, detection bbox/mask, depth.
너의 담당 출력: track_id별 dimensions와 measurement_status, 실측 오차 보고서.
탐지 담당에게는 동일 원본/프레임 ID/해상도/원본 bbox 좌표를 먼저 요청하면 됩니다.

## 근거 문서

- 카메라 투영/보정: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
- Depth Anything V2: https://github.com/DepthAnything/Depth-Anything-V2
- Metric 모델: https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth

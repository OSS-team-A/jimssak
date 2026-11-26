# 크기 보정 (일반 영상의 과대 추정 줄이기)

`ai.measurement.calibration`은 개발용 장면 측정 결과로 크기 보정 배율을 학습하고,
측정 결과에 적용한다. 모델 가중치 없이 실행된다.

## 왜 필요한가

개발용 ARKitScenes 장면 100개(가구 71개, 치수 213개)를 Depth Anything V2로 측정한 결과,
일반 영상(카메라 위치 없음, 웹 서비스와 같은 조건)은 크기를 일정하게 크게 냈다.

| 치수 | 추정/실제 중앙값 |
|---|---:|
| 가로 | 1.54 |
| 깊이 | 1.51 |
| 높이 | 1.19 |
| 전체 | 1.39 |

장면마다 차이는 있지만(10~90백분위 1.10~1.70) 거의 항상 크다. ARKit 영상은 여러 시점
결합이 되면 1.01로 치우침이 없고, 프레임별 중앙값으로 잰 경우만 1.50으로 크다.

## 방법

(영상 종류, 치수, 측정 방식)마다 `log(실제/추정)`의 중앙값으로 배율을 정한다. 표본이
10개 미만인 묶음은 (영상 종류, 치수), 그다음 영상 종류 전체의 배율을 쓴다. 길이를 아는
물건으로 이미 맞춘 영상(`reference_scaled`)은 건드리지 않는다. 보정한 물품에는
`size_calibrated` 표시가 붙는다.

`fit`은 방(ARKitScenes `visit_id`) 하나씩 빼고 나머지로 학습해 빠진 방을 채점한 결과도
함께 낸다. 같은 방의 다른 영상이 학습과 채점에 섞이지 않는다.

## 첫 결과

장면 하나씩 빼고 학습한 빠른 확인(같은 방의 다른 장면이 섞일 수 있음)에서 일반 영상
평균 절대오차가 다음과 같았다. 방 단위 결과는 아래 `fit` 출력으로 확인한다.

| 보정 | 평균 절대오차 |
|---|---:|
| 없음 | 25.3cm |
| 전체 한 배율 | 14.0cm |
| 치수별 | 12.7cm |
| 치수 + 측정 방식별 | 12.5cm |

## 맥에서 실행

학습 (몇 초):

```bash
python -m ai.measurement.calibration fit outputs/dev-eval-100/truth.json outputs/dev-eval-100/dav2.json --metadata outputs/arkit-depth-fix/metadata.csv --output ai/measurement/calibration/depth-anything-v2.json
```

출력의 `leave_one_room_out`에서 `raw`(보정 전)와 `calibrated`(보정 후)를 비교한다.
`within10`은 상대오차 10% 이내 비율이다.

측정 결과에 적용:

```bash
python -m ai.measurement.calibration apply outputs/result.json --calibration ai/measurement/calibration/depth-anything-v2.json --output outputs/result-calibrated.json
```

배율은 깊이 모델마다 다르므로 모델별로 따로 학습한다. 최종 평가용 장면은 학습과 배율
선택에 쓰지 않는다.

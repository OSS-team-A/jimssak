# 자유 이동 영상 크기 추정

`ai.measurement.moving_video`는 같은 영상에서 같은 물체의 프레임별 관측을 받아
센티미터 단위의 치수를 추정한다. 물체 검출과 추적은 호출부에서 담당한다.
한 작업에 여러 영상을 넣을 수 있다. 카메라 궤적, 추적 ID, 보정 배율은 영상마다 독립적이다.

## 실행

`.[vision,depth]` 의존성과 [Depth Anything V2 Metric Indoor Small 사전 학습 체크포인트](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf)의
`config.json`, `preprocessor_config.json`, `model.safetensors`가 필요하다.
가중치는 로컬 디렉터리에서 읽는다.

```bash
python -m pip install -e '.[vision,depth]'
python -m ai.measurement.moving_video /path/to/input.json \
  --model /path/to/local-checkpoint --device auto --output outputs/measurement.json
PYTHONPATH=. python experiments/size-estimation/score_moving_video.py \
  outputs/measurement.json /path/to/truth.json --output outputs/measurement-scores.json
```

`auto`는 사용 가능한 MPS, CUDA, CPU 순서로 깊이 추론 장치를 선택한다.
OpenCV 영상 디코딩과 기하 처리는 CPU에서 동작한다. `--device mps`는 GPU 실행을
명시할 때 쓴다.

## 물체 인식 모듈과 주고받는 데이터

[입력·출력 스키마](../ai/measurement/moving_contract.py)는
`MeasurementJob.model_json_schema()`, `MeasurementResponse.model_json_schema()`에서
JSON Schema로 확인할 수 있다.

- 요청은 `videos[]`다. 영상마다 `video_id`, `path`, `annotation_size_px`(가로, 세로),
  `views[]`를 넘긴다. 영상 경로는 요청 JSON 파일이 있는 폴더를 기준으로 해석한다.
- 관측마다 `frame_index`(0부터), `track_id`, `label`을 넘긴다.
  좌표는 영상의 회전 메타데이터를 반영한 뒤 `annotation_size_px`로 맞춘 이미지
  기준이다. 같은 물체의 ID를 모든 프레임에서 동일하게 보낸다.
- `bbox`는 `x1, y1, x2, y2` 형태다. 정확도를 높이려면
  `mask_polygon_px`(좌표 배열)나 `mask_rle`을 사용한다. RLE 길이는 행 우선 순서로
  배경부터 번갈아 기록하며 합이 이미지 픽셀 수와 같아야 한다. 첫 픽셀이 물체면
  첫 길이는 0이다. 화면 밖으로 잘렸거나 가린 경우에는 `complete: false`를 보낸다.
- 실험에서 상판을 따로 잴 때는 `tabletop_px`에 순서대로 네 점을 기록한다. 첫 변이
  상판 가로 방향이다. `floor_contacts_px`는 같은 물체의 바닥 접점, `overall_width_px`는
  실측 전체 폭의 양 끝에 대응하는 점이다. 가로·깊이는 바닥 평면의 긴 변·짧은 변이다.
- 보정 정보를 제공할 수 있으면 `intrinsics`에 `fx, fy, cx, cy`를 픽셀 단위로 보낸다.
  `camera_up`은 선택한 프레임 카메라 좌표에서의 위쪽 방향이다.
  `capture_poses[]`를 제공하면 모든 관측 프레임의 월드→카메라 회전과 미터 단위
  이동 벡터를 함께 보낸다. 좌표 변환은 `X_camera = R @ X_world + t`다.
- 결과는 `videos[].objects[]`의 물체별 `status`와 `dimensions`다. 치수마다
  `cm`, `method`, `scale_source`, `views`가 있다. 복원하지 못한 치수는 `null`이다.
  `registered_multiview`는 등록된 시점을 함께 쓴 값이고, `metric_depth_view_median`은
  개별 프레임에서 계산한 값의 중앙값이다. `poses[]`에서 연결된 구간, 사용한 특징점,
  재투영 오차와 추정 카메라 위치를 확인할 수 있다.
- `scale_reference`는 보정 실험에서만 넘긴다. 지정한 물체의 치수 하나로 영상 전체의
  추정값에 동일한 배율을 적용한다. 정답 파일은 추정기가 읽지 않으며, 평가 스크립트는
  보정에 사용한 치수를 오차 계산에서 제외한다.

[IMG_3898 관측 예제](examples/moving-img3898-input.json)와
[별도 실측 정답 파일](examples/moving-img3898-truth.json)은 표본 두 프레임을 사용한다.
예제 사본에서 영상 경로를 원본 위치에 맞춘다.

## 실행 검증

Mac M4에서 추가 학습 없이 Metric Indoor Small을 실행했다. `IMG_3898.MOV`의
SHA-256은 `b8b37cf7767e0690868d092b4bc9e390edf82ea2efcc100dba4b4da751d97c72`다.
170·196번 프레임에서 같은 흰색 의자 책상의 상판과 바닥 접점을 지정했다.
가정한 초점거리(540 × 960에서 611px)와 건물의 수직선으로 계산한 카메라의 위쪽
방향을 두 방식에 동일하게 사용했다. 책상 구간 추정 이동량은 약 4.5cm였다.
이 구간은 여러 시점을 3D로 합치기 위한 이동량이 부족해 프레임별 중앙값을 사용했다.

| 치수 | 실측(cm) | 치수 보정 없이 추정 · 절대오차(cm) | 상판 가로만 보정 · 절대오차(cm) |
|---|---:|---:|---:|
| 상판 가로 | 59 | 104.1 · 45.1 | 59.0 · 보정 입력 |
| 상판 깊이 | 39 | 64.4 · 25.4 | 36.5 · 2.5 |
| 상판에서 바닥까지 | 73 | 118.4 · 45.4 | 67.1 · 5.9 |
| 전체 가로 | 95 | 측정 불가 | 측정 불가 |

95cm는 영상의 어떤 두 점이 실측 양 끝에 대응하는지 확인되지 않아 채점할
추정값을 산출하지 않았다. 보정 후 깊이와 높이는 같은 물체의 미사용 치수 평가다.

별도로 [ARKitScenes](https://machinelearning.apple.com/research/arkitscenes)의
실제 이동 촬영 `48458732`에서 원본 RGB를 시간순으로 재생하고, 테이블 마스크가
있는 3프레임을 측정했다. 동일한 사전 학습 모델, 카메라 내부 정보, 위쪽 방향을
사용했다. 3D 주석에 기록된 가구 치수는 채점에만 썼다. RGB 자세 추정에서
카메라의 처음·마지막 위치 간 거리는 61.1cm였고, 촬영 궤적 기준은 48.3cm였다.

| 입력과 시점 결합 | 가로·깊이·높이 추정(cm) | 절대오차(cm) |
|---|---|---|
| 마스크, RGB로 위치 추정 후 단일 프레임 값 집계 | 208.4 / 106.3 / 64.1 | 134.4 / 56.9 / 21.5 |
| 사각형, RGB로 위치 추정 후 단일 프레임 값 집계 | 124.5 / 120.7 / 66.9 | 50.5 / 71.2 / 24.2 |
| 마스크, 촬영 기기가 기록한 위치로 여러 시점 결합 | 80.4 / 49.4 / 35.1 | 6.3 / 0.1 / 7.5 |
| 참고용 3D 주석 | 74.0 / 49.5 / 42.6 | — |

원본 책상 프레임의 깊이 추론 중앙값은 MPS 0.139초, CPU 0.322초였다. 한 번 실행한
6프레임 기준 약 2.3배 차이이며, 디코딩과 카메라 복원은 이 시간에 포함하지 않는다.
이번 검증에서 두 장치의 치수는 0.01cm 단위에서 일치했다.

일반 RGB 영상에서는 사전 학습 모델이 제공하는 미터 스케일과 카메라 위치가 모두
추정값이다. 반복 무늬, 시점 간 이동 부족, 가림, 초광각 왜곡에서 오차가 커진다.
다음 검증에서는 같은 물품을 여러 방향에서 촬영하고 ARKit의 포즈와 내부 정보를
함께 저장해 RGB 추정 오차와 치수 계산 오차를 분리한다.

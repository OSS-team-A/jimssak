# 영상에서 책상·의자 검출 및 치수 추정, 중복 검토

저장소 루트에서 실행합니다. 실험은 로컬 원본을 읽으며, 추론에 사용할
[YOLO11n 가중치](https://github.com/ultralytics/assets/releases/tag/v8.3.0)는 별도로 받습니다.
첫 실행 환경은 macOS arm64, Python 3.12입니다. 실험 전용 환경을 분리하면
기존 OpenCV headless 환경과 충돌을 피할 수 있습니다.

```bash
python3.12 -m venv outputs/venv-video
source outputs/venv-video/bin/activate
python -m pip install -r ai/training/requirements-video.txt
export YOLO_AUTOINSTALL=false

VIDEO=/path/to/video.MOV
WEIGHTS=/path/to/yolo11n.pt
OUT=outputs/video-experiment

python -m ai.training.video_inventory "$VIDEO" --weights "$WEIGHTS" --output "$OUT/inventory"
python -m ai.measurement.video_geometry "$VIDEO" --annotations /path/to/annotations.json \
  --output "$OUT/geometry"
python -m ai.training.video_review "$OUT/inventory" --annotations /path/to/identity-review.json \
  --output "$OUT/review"
```

- `video_inventory`는 표시 방향으로 6프레임마다 이미지를 저장하고, COCO 클래스 56과 60을
  검출합니다. 기본 신뢰도 하한은 0.15, 추론 입력 크기는 960, NMS IoU는 0.5입니다.
  클래스별 BoT-SORT는 특징점 기반 카메라 움직임 보정을 사용합니다. CSV에는 모든 검출과
  할당된 트랙 ID가 담깁니다. `inventory.json`은 원본·가중치 SHA-256과 실행 버전을 기록합니다.
  출력 디렉터리에는 새 경로를 사용합니다. 클래스명 `dining table`은 일반 상판 검출 후보로 해석합니다.
- `video_geometry` 입력의 `annotation_size_px`는 주석을 단 이미지의 `[너비, 높이]`입니다.
  `reference_cm`에 `top_width`, `top_depth`, `top_to_floor`, `overall_span_reported`를 적습니다.
  `references`에는 `frame`, 동일 물체에 공통인 `unit`, 너비 변부터 시작해 둘레를 순서대로
  따라가는 `top` 네 점을 적습니다. 같은 높이라고 가정할 수 있는 비교 상판은
  `comparison_top`, 동일한 바닥 레일 양 끝은 `floor_rail`로 선택 사항입니다.
  `focal_candidates_px_at_source_size`에는 원본 표시 해상도의 초점거리 가정 목록을 적고,
  `jitter_px_at_source_size`에는 좌표 민감도 계산용 오차 표준편차를 적습니다.
  `motion_pairs`는 특징점을 비교할 0부터 시작하는 프레임 번호 쌍입니다.
  `source_sha256`을 넣으면 영상이 주석과 일치하는지 검사합니다.
- `video_review` 입력에는 원본·모델 해시 `source_sha256`, `model_sha256`, `sample_stride`와
  영상에서 직접 대응을 확인한 `verified_units`를 적습니다. 각 항목에는 `unit_id`,
  `first_frame`, `last_frame`, 포함되는 검출 트랙 ID 배열 `tracks`가 필요합니다.
  `single_frame_chair_census`에는 `frame`, `coordinate_size_px`,
  `chair_back_points`를 적습니다. 결과 CSV는 각 검증 구간에서 프레임당 물체 하나로
  합친 상자와 원래 트랙들을 기록합니다.

트랙 수는 가림, 화면 이탈, 같은 상판의 부분 검출 영향을 받습니다. 전체 고유 개수를 확정하려면
영상 전 구간에서 물리적 정체를 대조해야 합니다. 상판 호모그래피 수치는 공면 가정에 의존합니다.
포즈와 바닥 길이 계산은 카메라 내부 파라미터 가정, 상판 모델, 상판·바닥 높이 차이에 의존합니다.
95cm와 같은 외형 치수를 비교하려면 영상 점 두 개가 실측에 사용한 기준점과 일치해야 합니다.

원본 MOV, 주석, 검출 이미지, 모델 가중치, 실행 결과는 로컬에 보관합니다.

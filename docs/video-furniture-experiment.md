# 영상에서 책상·의자 검출 및 중복 검토

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
python -m ai.training.video_review "$OUT/inventory" --annotations /path/to/identity-review.json \
  --output "$OUT/review"
```

- `video_inventory`는 표시 방향으로 6프레임마다 이미지를 저장하고, COCO 클래스 56과 60을
  검출합니다. 기본 신뢰도 하한은 0.15, 추론 입력 크기는 960, NMS IoU는 0.5입니다.
  클래스별 BoT-SORT는 특징점 기반 카메라 움직임 보정을 사용합니다. CSV에는 모든 검출과
  할당된 트랙 ID가 담깁니다. `inventory.json`은 원본·가중치 SHA-256과 실행 버전을 기록합니다.
  출력 디렉터리에는 새 경로를 사용합니다. 클래스명 `dining table`은 일반 상판 검출 후보로 해석합니다.
- `video_review` 입력에는 원본·모델 해시 `source_sha256`, `model_sha256`, `sample_stride`와
  영상에서 직접 대응을 확인한 `verified_units`를 적습니다. 각 항목에는 `unit_id`,
  `first_frame`, `last_frame`, 포함되는 검출 트랙 ID 배열 `tracks`가 필요합니다.
  `single_frame_chair_census`에는 `frame`, `coordinate_size_px`,
  `chair_back_points`를 적습니다. 결과 CSV는 각 검증 구간에서 프레임당 물체 하나로
  합친 상자와 원래 트랙들을 기록합니다.

트랙 수는 가림, 화면 이탈, 같은 상판의 부분 검출 영향을 받습니다. 전체 고유 개수를 확정하려면
영상 전 구간에서 물리적 정체를 대조해야 합니다.

원본 MOV, 주석, 검출 이미지, 모델 가중치, 실행 결과는 로컬에 보관합니다.

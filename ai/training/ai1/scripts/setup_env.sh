#!/usr/bin/env bash
# 환경 창(컨테이너 안)에서 실행: 기존 PyTorch를 바꾸지 않고 프로젝트 의존성을 설치한다.
#   cd /workspace/jimssak && bash scripts/setup_env.sh
# 호스트(서버 창)에서 실행하지 않는다. sudo / apt / 드라이버 변경을 하지 않는다.
set -euo pipefail
cd "$(dirname "$0")/.."

CONSTRAINTS="$(mktemp /tmp/jimssak_torch_constraints.XXXXXX.txt)"
python - "$CONSTRAINTS" <<'PY'
import sys, importlib.metadata as m
lines = []
for p in ("torch", "torchvision", "torchaudio"):
    try:
        lines.append(f"{p}=={m.version(p)}")
    except m.PackageNotFoundError:
        pass
if not any(l.startswith("torch==") for l in lines):
    sys.exit("오류: 이 환경에 torch가 없습니다. 기본 이미지(pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel) 환경에서 실행하세요.")
open(sys.argv[1], "w").write("\n".join(lines) + "\n")
print("고정할 기존 패키지:", ", ".join(lines))
PY

BEFORE="$(python -c 'import torch; print(torch.__version__, torch.version.cuda)')"
python -m pip install -r requirements.txt -c "$CONSTRAINTS"
AFTER="$(python -c 'import torch; print(torch.__version__, torch.version.cuda)')"
if [ "$BEFORE" != "$AFTER" ]; then
  echo "오류: 설치 중 torch가 바뀌었습니다 ($BEFORE → $AFTER). 설치 로그를 확인하세요." >&2
  exit 1
fi
echo "torch 유지 확인: $AFTER"

# opencv-python(GUI 빌드)이 libGL 부족으로 import 실패하면 같은 버전의 headless 빌드로 교체
if ! python -c "import cv2" 2>/dev/null; then
  CV_VER="$(python -c 'import importlib.metadata as m; print(m.version("opencv-python"))')"
  echo "cv2 import 실패 → opencv-python-headless==$CV_VER 로 교체합니다"
  python -m pip uninstall -y opencv-python
  python -m pip install "opencv-python-headless==$CV_VER" -c "$CONSTRAINTS"
fi
python -c "import cv2, ultralytics; print('cv2', cv2.__version__, '| ultralytics', ultralytics.__version__)"
rm -f "$CONSTRAINTS"
echo "설치 완료. 다음: python scripts/check_env.py --require-gpu --yolo --video <영상 경로>"

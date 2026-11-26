#!/usr/bin/env bash
# 짐싹 전체 자동 실행 (환경 창/컨테이너 안에서, 보통 서버 창의 lab-env run 으로 호출)
#
#   lab-env run jimssak "cd /workspace/jimssak && bash scripts/run_all.sh --profile standard"
#
# 옵션
#   --profile smoke|standard|full   (기본 standard)
#   --skip-smoke                    standard/full 전에 smoke 점검을 생략 (기본: smoke 먼저 수행)
#   --resume <run 이름|latest>      실패한 실행을 이어서 (성공한 단계는 건너뜀)
#   --from-stage <단계>             해당 단계부터 다시
#   --allow-cpu                     GPU 없이 점검 (서버 본 실행에서는 쓰지 않음)
#
# 서버 설정(sudo/apt/드라이버/Docker)은 바꾸지 않는다. 패키지는 프로젝트 전용 가상환경(.venv)에만 설치하고,
# 기본 이미지의 torch/torchvision은 --system-site-packages 로 그대로 재사용한다(버전 고정·변경 시 중단).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

PROFILE=standard; SKIP_SMOKE=0; RESUME=""; FROM=""; ALLOW_CPU=""
while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2;;
    --skip-smoke) SKIP_SMOKE=1; shift;;
    --resume) RESUME="$2"; shift 2;;
    --from-stage) FROM="$2"; shift 2;;
    --allow-cpu) ALLOW_CPU="--allow-cpu"; shift;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2;;
  esac
done

STAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p logs
MAIN_LOG="logs/run_all_${PROFILE}_${STAMP}.log"
exec > >(tee -a "$MAIN_LOG") 2>&1
echo "==== 짐싹 run_all 시작 $(date '+%F %T') | profile=$PROFILE | 로그 $ROOT/$MAIN_LOG ===="

# 캐시는 영속 저장소(/workspace)에
if [ -d /workspace ] && [ -w /workspace ]; then
  export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
  export TORCH_HOME="${TORCH_HOME:-/workspace/.cache/torch}"
  export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/workspace/.cache/ultralytics}"
else
  export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
  export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
fi
export PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=False

SYS_PY="$(command -v python3 || command -v python)"
export JIMSSAK_TORCH_BEFORE="$("$SYS_PY" -c 'import torch; print(torch.__version__, torch.version.cuda)')"
echo "기존 torch: $JIMSSAK_TORCH_BEFORE"

# ---- 1~2단계 일부: 전용 가상환경 + 의존성 설치 ----
if [ ! -x .venv/bin/python ]; then
  echo "[setup] 전용 가상환경 생성(.venv, --system-site-packages 로 기존 torch 재사용)"
  if ! "$SYS_PY" -m venv --system-site-packages .venv; then
    echo "[setup] ensurepip 없이 가상환경 생성 후 기존 pip로 설치합니다"
    rm -rf .venv
    "$SYS_PY" -m venv --system-site-packages --without-pip .venv
  fi
fi
PY="$ROOT/.venv/bin/python"
# 가상환경 안에 pip가 없으면 기존 파이썬의 pip로 이 가상환경에 설치 (pip --python, pip 22.3+)
if "$PY" -m pip --version >/dev/null 2>&1; then
  pipi() { "$PY" -m pip "$@"; }
else
  pipi() { "$SYS_PY" -m pip --python "$PY" "$@"; }
fi
CONS="$(mktemp --suffix=.txt)"
"$PY" - "$CONS" <<'PYEOF'
import sys, importlib.metadata as m
out = []
for p in ("torch", "torchvision", "torchaudio"):
    try: out.append(f"{p}=={m.version(p)}")
    except m.PackageNotFoundError: pass
open(sys.argv[1], "w").write("\n".join(out) + "\n"); print("고정:", out)
PYEOF
if [ ! -f .venv/.deps_ok ] || [ requirements.txt -nt .venv/.deps_ok ]; then
  echo "[setup] 의존성 설치 (requirements.txt, torch 고정)"
  pipi install -r requirements.txt -c "$CONS"
  if ! "$PY" -c "import cv2" 2>/dev/null; then
    CV="$("$PY" -c 'import importlib.metadata as m; print(m.version("opencv-python"))')"
    echo "[setup] cv2 import 실패(libGL) → opencv-python-headless==$CV"
    pipi uninstall -y opencv-python
    pipi install "opencv-python-headless==$CV" -c "$CONS"
  fi
  touch .venv/.deps_ok
fi
AFTER="$("$PY" -c 'import torch; print(torch.__version__, torch.version.cuda)')"
if [ "$AFTER" != "$JIMSSAK_TORCH_BEFORE" ]; then
  echo "오류: 설치 후 torch가 바뀌었습니다 ($JIMSSAK_TORCH_BEFORE → $AFTER). 중단합니다." >&2; exit 1
fi
rm -f "$CONS"

ARGS=($ALLOW_CPU)
[ -n "$FROM" ] && ARGS+=(--from-stage "$FROM")

if [ -n "$RESUME" ]; then
  "$PY" -u run_pipeline.py --profile "$PROFILE" --resume "$RESUME" "${ARGS[@]}"
else
  if [ "$PROFILE" != "smoke" ] && [ "$SKIP_SMOKE" != "1" ]; then
    echo "==== [1/2] smoke 점검 실행 (전체 단계가 끝까지 도는지 먼저 확인) ===="
    "$PY" -u run_pipeline.py --profile smoke "${ARGS[@]}"
    echo "==== smoke 점검 통과 → [2/2] $PROFILE 본 실행 ===="
  fi
  "$PY" -u run_pipeline.py --profile "$PROFILE" "${ARGS[@]}"
fi

echo "==== 전체 완료 $(date '+%F %T') ===="
ls -lh outputs/jimssak_results_*.tar.gz 2>/dev/null | tail -n 3
echo "돌려줄 파일: 위 목록의 가장 최근 jimssak_results_${PROFILE}_*.tar.gz (전체 로그: $ROOT/$MAIN_LOG)"

#!/usr/bin/env bash
# B0 학습 전 기준선만 (scripts/run_all.sh 로 설치·가상환경이 준비된 뒤 사용)
#   bash scripts/run_baseline.sh --profile standard [--resume <run 이름|latest>] [--allow-cpu]
# --resume 를 생략하면 latest(같은 프로필의 가장 최근 실행)를 이어서 사용한다.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; [ -x "$PY" ] || { echo "먼저 bash scripts/run_all.sh 를 실행해 가상환경을 만드세요" >&2; exit 1; }
PROFILE=standard; RESUME=latest; EXTRA=()
while [ $# -gt 0 ]; do case "$1" in
  --profile) PROFILE="$2"; shift 2;; --resume) RESUME="$2"; shift 2;; --allow-cpu) EXTRA+=(--allow-cpu); shift;;
  *) echo "알 수 없는 옵션 $1" >&2; exit 2;; esac; done
export PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=False
if [ "$RESUME" = latest ] && ! ls outputs/${PROFILE}_*/state.json >/dev/null 2>&1; then RESUME_ARGS=(); else RESUME_ARGS=(--resume "$RESUME"); fi
mkdir -p logs
"$PY" -u run_pipeline.py --profile "$PROFILE" "${RESUME_ARGS[@]}" --only env_check deps_check validate_inputs pretrained_download baseline_b0_video "${EXTRA[@]}" 2>&1 | tee -a "logs/run_baseline_$(date +%Y%m%d-%H%M%S).log"

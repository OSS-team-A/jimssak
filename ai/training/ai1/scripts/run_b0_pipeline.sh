#!/usr/bin/env bash
# B0 기준선 전체 실행을 한 번에: 설치 → GPU 확인 → 추론 → 재등장 연결 → 검수표 → 시간 요약 → 결과 압축
#
# 서버 창에서 (긴 작업이므로 lab-env run 사용):
#   lab-env run <환경명> "cd /workspace/jimssak && bash scripts/run_b0_pipeline.sh /workspace/jimssak_data/videos/unnie_room_01.mp4 unnie_room_01"
#
# 인자: <영상 경로> [video_id, 기본 unnie_room_01]
# 환경변수(선택): DEVICE=cuda:0 (기본) | SKIP_SETUP=1 (설치 생략)
# 끝나면 /workspace/jimssak_results_<video_id>_<시각>.tgz 하나가 생긴다. 이 파일을 코드 작성자에게 돌려준다.
# 서버 설정(sudo, apt, 드라이버, Docker)은 바꾸지 않는다. pip 설치는 이 컨테이너 안에만 적용된다.
set -euo pipefail
cd "$(dirname "$0")/.."

VIDEO="${1:?사용법: bash scripts/run_b0_pipeline.sh <영상 경로> [video_id]}"
VID="${2:-unnie_room_01}"
DEVICE="${DEVICE:-cuda:0}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN="gpu_${VID}_${STAMP}"
[ "${DEVICE#cuda}" = "$DEVICE" ] && RUN="${DEVICE}_${VID}_${STAMP}"
OUT_PARENT="$(cd .. && pwd)"
RESULT="${OUT_PARENT}/jimssak_results_${VID}_${STAMP}.tgz"

step() { echo; echo "==== [$(date +%H:%M:%S)] $* ===="; }

if [ ! -f "$VIDEO" ]; then
  echo "오류: 영상 파일이 없습니다: $VIDEO" >&2
  exit 2
fi

step "0. 정보: 영상=$VIDEO video_id=$VID device=$DEVICE run=$RUN"
python --version

if [ "${SKIP_SETUP:-0}" != "1" ]; then
  step "1. 패키지 설치 (기존 torch 유지)"
  bash scripts/setup_env.sh
else
  step "1. 패키지 설치 생략 (SKIP_SETUP=1)"
fi

step "2. 환경·GPU 확인 (실패하면 여기서 멈춤)"
mkdir -p runs evaluations
REQ=""
[ "${DEVICE#cuda}" != "$DEVICE" ] && REQ="--require-gpu"
python scripts/check_env.py $REQ --device "$DEVICE" --yolo --video "$VIDEO" --out "runs/env_check_${STAMP}.json"

step "3. B0 추론 (YOLO11n-seg + BoT-SORT)"
python -u infer.py --video "$VIDEO" --video-id "$VID" --label B0 --device "$DEVICE" --run-name "$RUN"

step "4. 재등장 연결 (B2 후처리)"
python -u link_reappearance.py --run "runs/B0/$RUN" --out-name relink_v1

step "5. 사람이 채울 매칭 검수표 생성"
python make_review_sheet.py --items "runs/B0/$RUN/items.csv" --out "annotations/match_review_B0_${RUN}.csv"

step "6. 처리 시간 요약"
python summarize_runs.py --runs "runs/B0/$RUN" --out "evaluations/timing_${RUN}.csv"

step "7. 재등장 장면 그림 (해당 쌍이 없으면 건너뜀)"
python make_reappearance_figure.py --run "runs/B0/$RUN" --relink "runs/B0/$RUN/relink_v1" \
  || echo "재등장 후보 쌍이 없어 그림을 만들지 않았습니다 (오류 아님)"

step "8. 결과 압축"
tar -czf "$RESULT" "runs/B0/$RUN" "runs/env_check_${STAMP}.json" \
  "annotations/match_review_B0_${RUN}.csv" "evaluations/timing_${RUN}.csv" "evaluations/timing_${RUN}.json"
ls -lh "$RESULT"

echo
echo "==== 전체 완료 ===="
echo "결과 폴더: $(pwd)/runs/B0/$RUN"
echo "돌려줄 파일: $RESULT"

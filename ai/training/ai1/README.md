# 짐싹 AI — 이사 물품 탐지·추적·재식별·분실/파손 판단 자동 학습·평가 파이프라인

방 영상에서 이사 물품을 찾고(인스턴스 세그멘테이션), 같은 물품의 재등장을 한 번만 세고, 이사 전후 영상을 비교해
분실 의심·손상 변화를 표시하는 AI 모듈과, 이를 **공개 데이터로 자동 학습·평가**하는 파이프라인이다.

- **서버 실행 담당자**: [HANDOFF.md](HANDOFF.md) 만 보면 된다 (명령 한 줄).
- 결과 해석: [docs/results_guide.md](docs/results_guide.md) · 오류 대응: [docs/troubleshooting.md](docs/troubleshooting.md)
- 데이터셋·모델 선정 근거: [docs/model_and_data_selection.md](docs/model_and_data_selection.md)
- 서버 코드 작성 지침: [AGENT.md](AGENT.md) · 사람이 작성하는 정답·검수: [docs/labeling_guide.md](docs/labeling_guide.md)

## 실험 단계

| 단계 | 내용 | 학습 | 평가 데이터 |
|---|---|---|---|
| **B0** | COCO 사전학습 YOLO11n-seg + 기본 BoT-SORT (학습 전 기준선) | 없음 | LVIS 테스트(COCO val2017), 언니방영상(정답 없음 → 검수표) |
| **B1** | 이사 물품 클래스로 파인튜닝한 YOLO11n/s(-m)-seg, 실내 탐지 YOLO11s | LVIS 마스크, HomeObjects 박스 | LVIS 테스트(공통/신규 클래스 분리), HomeObjects val |
| **B2** | 재식별 기반 재등장 연결 (색 히스토그램 → DINOv2 → triplet 학습 헤드), BoT-SORT vs ByteTrack | ABO 다중 시점(triplet) | ABO 테스트 검색, 합성 카메라 이동 영상(HOTA·IDF1·중복·잘못된 병합) |
| **B3** | 이사 전후 매칭·분실 의심 (임베딩 + 코사인 + Hungarian, 임계값은 검증으로 선택) | 임계값 선택 | ABO 합성 전후 시나리오 (실제 영상 평가는 미측정) |
| **B4** | 손상: 균열 YOLO-seg 파인튜닝, PatchCore 이상 탐지, 정합+특징 차이 전후 변화 | crack-seg, VisA 정상, (메모리) | crack-seg 테스트(실제), VisA 테스트(실제·산업), ABO 합성 손상 |

## 실행

```bash
# [서버 창] — 전 과정 자동 (smoke 점검 → standard)
lab-env run jimssak "cd /workspace/jimssak && bash scripts/run_all.sh --profile standard"
# 재개
lab-env run jimssak "cd /workspace/jimssak && bash scripts/run_all.sh --profile standard --resume latest"
```
결과: `outputs/jimssak_results_<프로필>_<시각>.tar.gz` (보고서·비교표·그래프·가중치·API JSON·로그)

프로필: `configs/profiles/{smoke,standard,full,max}.yaml` (데이터 양·모델 크기·epoch·batch). 규모·디스크 추정은 HANDOFF.md 표.
`max`는 standard 결과 분석을 반영한 정확도 개선판이다: 공통 품목 라벨을 COCO 원본(빠짐없는 라벨)으로 교체,
조건에 맞는 LVIS 학습 이미지 전부 사용, YOLO11m-seg 추가, GPU 2장 DDP, DINOv2 ViT-B 재식별 백본.
전처리 결과 폴더는 데이터 관련 설정의 해시(`data/datasets/<프로필>_<해시>`)로 구분되어 설정을 바꾸면 새로 만들어진다.

## 폴더

```text
jimssak/
├── HANDOFF.md / README.md / AGENT.md / requirements.txt
├── run_pipeline.py            # 오케스트레이터 진입점 (단계 실행·기록·재개)
├── infer.py                   # 영상 1편 추론 (YOLO-seg + BoT-SORT, 모든 검출 + 임시 track_id)
├── link_reappearance.py       # 색 히스토그램 규칙 재등장 연결
├── compare_moves.py           # 실제 이사 전후 영상 비교 (분실 의심·추가·수량)
├── evaluate_list.py / evaluate_frames.py / make_review_sheet.py / train.py / ...   # 단독 실행 도구
├── configs/                   # taxonomy(통합 클래스), datasets(출처·라이선스), profiles, pipeline_default
├── scripts/                   # run_all.sh(최상위), setup_env.sh, check_env.py, download_datasets.sh, run_baseline.sh,
│                              # run_training.sh, run_evaluation.sh, package_results.sh, run_b0_pipeline.sh(B0만)
├── src/jimssak_ai1/
│   ├── orchestrator.py        # 19단계 정의
│   ├── datasets/              # 다운로드(이어받기·재시도), LVIS·HomeObjects·crack-seg·VisA·ABO 전처리
│   ├── detection/             # YOLO 학습 래퍼(OOM·DDP·resume), 탐지·분할 평가(연합 주석·혼동행렬·IoU·Dice)
│   ├── tracking/              # 합성 영상 생성, HOTA/IDF1/CLEAR, 추적·재등장 벤치
│   ├── reidentification/      # 임베딩(hist·DINOv2·학습 헤드), triplet 학습, 검색 지표
│   ├── matching/              # 전후 Hungarian 매칭·분실 의심 상태, 합성 시나리오 평가
│   ├── damage/                # PatchCore, 합성 손상, 정합·변화 지도·상태 판정
│   ├── pipeline.py / video_items.py   # 영상 추론, API JSON
│   └── compare_tables.py / report.py  # 비교표·그래프·보고서
├── data/input_videos/언니방영상.mp4   # 입력 테스트 영상 (학습에 사용하지 않음)
├── data/{cache,datasets}/     # 실행 시 자동 다운로드 (전달 압축에 미포함)
├── models/pretrained/         # 실행 시 자동 다운로드
├── outputs/  logs/            # 실행 결과·로그
├── annotations/ templates/    # 사람이 채우는 정답·검수표
├── docs/  tests/
```

## 검증 범위 (정직한 상태)

- **개발 PC(Windows, CPU, Intel Arc — CUDA 없음)에서 smoke 프로필 전 단계를 `--allow-cpu`로 실제 실행**해 코드 경로를 확인했다
  (결과 수치는 smoke 소량 데이터라 성능 근거가 아님). 상세는 마지막 실행 보고를 참고.
- **서버 GPU(RTX PRO 4500 Blackwell)에서의 실행·DDP·처리 시간·standard/full 결과는 미검증**이다. 서버 실행이 첫 검증이다.
- 실제 이사 전후 영상 기반 분실 판단, 실제 가구 손상 탐지, 언니방영상 정확도, 실제 영상 추적 HOTA는 **정답 데이터가 없어 미측정**으로 보고된다.

## API 출력 (`api/items_*.json`, schema `jimssak.item.v1`)

물품별: `item_id, tracking_id, linked_tracking_ids, class_name, confidence, first_seen_timestamp, last_seen_timestamp,
representative_image, bounding_box, segmentation_mask(polygons), reid_embedding_reference, matched_item_id, matching_confidence,
missing_status, damage_status, damage_type, damage_region, damage_confidence, dimensions_cm(null), box_fit_60x40x40("unknown"),
review_required, review_reason`. 치수·부피는 구현되지 않아 null이며, 박스 적합은 판단 보류다.

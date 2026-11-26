# 결과 해석 안내 (결과 압축을 받은 뒤 프로젝트 담당자가 볼 것)

압축을 풀면 `<run 이름>/` 하나가 나온다(예: `standard_20261126-101500/`). 먼저 볼 순서:

1. `report/REPORT.md` — 전체 요약 보고서(측정값만, 미측정 항목과 사유 포함). 그림은 같은 폴더 PNG.
2. `stage_status.csv` — 단계별 성공/실패·소요 시간. 하나라도 failed면 그 뒤 결과는 없다.
3. `compare/` — 발표용 비교표 CSV
4. `api/items_B2_final.json` — 언니방영상의 최종 물품 목록(백엔드 연동 형식)

## 폴더 구성

| 경로 | 내용 |
|---|---|
| `env/` | 서버 GPU·CUDA·PyTorch·패키지 버전(`env.json`, `packages.json`), 사전학습 모델 목록, 학습 시간 추정(`time_estimate.json`) |
| `datasets/` | 다운로드 기록(`downloads.jsonl`), 무결성(`integrity.csv`), 라이선스(`licenses.csv`), 클래스 확보 현황(`class_availability.csv`), 데이터 통계, 분할 누수 검사(`split_report.json`) |
| `models/yolo/<이름>/` | YOLO 학습 기록: `results.csv`(epoch별 loss·P·R·mAP), `args.yaml`, 학습 그래프, `jimssak_train_done.json`(학습 시간·배치·재시도) |
| `models/reid/` | 재식별 헤드 학습 기록 `history.csv`, `train_summary.json` |
| `final_models/` | 학습된 가중치: `B1_yolo11*-seg_best.pt`(이사 물품 세그멘테이션), `B1_homeobjects_det_best.pt`, `B4_crack_seg_best.pt`, `B2_reid_head_best.pt` |
| `eval/detection/lvis_test/<모델>/` | 테스트(COCO val2017 이미지) 클래스별 AP·P/R/F1·FP·FN·IoU·Dice(`per_class.csv`), 혼동행렬, 요약 |
| `eval/detection/homeobjects_test/` | 실내 도메인 박스 평가 |
| `eval/tracking/` | 합성 영상 추적: 시퀀스별(`test_rows.csv`)·요약(`tracking_comparison.csv`), 임계값 선택 근거 |
| `eval/reid/` | 재식별 검색 지표 |
| `eval/matching/` | B3 전후 매칭(합성) |
| `eval/damage/` | B4: 균열(실제)·VisA(실제, 산업)·ABO 합성 이상탐지·전후 변화(합성) |
| `video/B0/B0`, `video/B1/B1` | 언니방영상 학습 전·후 분석 결과(결과 영상, 프레임별 JSONL, 후보 목록, 대표 사진) |
| `video/match_review_B*.csv` | 사람이 채울 매칭 검수표 |
| `api/` | 백엔드 연동 JSON (B0, B1, B2 최종) |
| `logs/` | GPU 사용량 기록, 셸 로그 |

## 표 읽는 법

### `compare/detection_before_after.csv`
- `subset=common_classes`: B0(COCO)도 원래 알던 클래스 → **파인튜닝으로 기존 품목 성능이 떨어지지 않았는지**
- `subset=new_classes`: B0가 아예 모르던 클래스 → B0 값은 구조적으로 0(미지원). **새 품목 지원 효과**
- `subset=all_gt_classes`: 서비스 관점 전체(미지원 품목도 정답에 포함)
- `FP`/`FN`: 운영 임계값(conf≥0.25, IoU≥0.5) 기준 오탐·누락 수. LVIS 연합 주석 규칙으로 확인되지 않은 클래스의 예측은 세지 않음

### `compare/ablation_service_synthetic_video.csv`
한 단계씩 더했을 때 서비스 지표가 어떻게 변했는지(합성 영상). `duplicate_rate`(같은 물건 두 번 셈)는 낮을수록,
`wrong_merge_rate`(다른 물건을 하나로 합침)도 낮을수록, `reappearance_link_accuracy`·`item_f1`·`IDF1`·`HOTA`는 높을수록 좋다.
**중복을 줄였는데 잘못된 병합이 늘었다면 개선이 아니다** — 둘을 같이 본다.

### B3 (`eval/matching/b3_comparison.csv`)
`missing_f1`(분실 의심 찾기), `false_missing_rate`(멀쩡한 물건을 분실로 잘못 표시한 비율), `match_accuracy`(전후 같은 물건을 맞게 짝지은 비율).
`count_only`는 수량만 비교하는 기준선이다. 실제 이사 전후 영상 평가는 미측정.

### B4 (`eval/damage/b4_eval.json`)
- `real_data.crack_segmentation`: 실제 균열 데이터의 학습 전(COCO, crack 미지원)·후 비교 — 도메인은 도로·벽
- `real_data.visa_patchcore`: 실제 산업 이상 데이터의 PatchCore vs 전역 kNN — 가구 아님
- `synthetic_data.*`: 합성 손상. 실제 성능 근거로 쓰지 않는다

## 발표에 쓸 때 주의
- smoke 결과는 성능 근거가 아니다(파이프라인 점검용).
- 언니방영상은 정답 목록이 없어 정확도 미측정 — 검수표를 채우고 `evaluate_list.py`로 채점해야 수치가 생긴다.
- 독립 평가 방·영상 수가 적으면 '파일럿에서 관찰한 차이'로 표현하고 통계적 유의성을 주장하지 않는다.
- 합성 실험(추적·B3·B4 변화)과 실제 공개 데이터 실험(탐지·재식별 검색·균열·VisA)을 섞어 말하지 않는다.

## 이사 전후 실제 영상이 생기면
두 영상을 각각 `infer.py --weights final_models/B1_..._best.pt`로 분석한 뒤:
```
python compare_moves.py --before <이전 분석 폴더> --after <이후 분석 폴더> \
    --embedder trained --reid-ckpt final_models/B2_reid_head_best.pt --threshold <b3_eval.json의 trained threshold> --out outputs/move_01
```

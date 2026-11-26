# AI 1 학습·평가 파이프라인 (물품 탐지·세그멘테이션·추적·재식별·전후 매칭·손상)

코드: `ai/training/ai1/` (자체 README·설정·스크립트를 가진 하위 프로젝트). AI 2 연동 안내: `docs/ai1-handoff-for-ai2.md`.

## 무엇을 하나

| 단계 | 내용 | 학습 | 평가 |
|---|---|---|---|
| B0 | COCO 사전학습 YOLO11n-seg + BoT-SORT (학습 전 기준선) | 없음 | LVIS 테스트 |
| B1 | 이사 물품 23~25종 세그멘테이션 파인튜닝 (YOLO11n/s/m-seg), 실내 박스 탐지(YOLO11) | LVIS·COCO 마스크, HomeObjects 박스 | LVIS 독립 테스트(공통/신규 품목 분리), HomeObjects |
| B2 | 재식별 기반 재등장 연결 (DINOv2 + triplet 학습 헤드), BoT-SORT vs ByteTrack | ABO 다중 시점 상품 사진 | ABO 검색, 합성 카메라 영상(HOTA·IDF1·중복·잘못된 병합) |
| B3 | 이사 전후 1:1 매칭(Hungarian)·분실 의심 | 임계값 검증 선택 | ABO 합성 시나리오 |
| B4 | 균열 세그멘테이션, PatchCore 이상 탐지, 전후 상태 변화 | crack-seg, VisA | crack-seg·VisA 테스트, 합성 손상 |

## 데이터 (모두 인증 없이 자동 다운로드, 레포에는 포함하지 않음)

| 데이터 | 출처 | 라이선스 | 용도 |
|---|---|---|---|
| LVIS v1 + COCO 2017 | lvisdataset.org, cocodataset.org | 주석 CC BY 4.0, 이미지별 Flickr 라이선스(비상업 포함) | 물품 마스크 학습·평가 |
| HomeObjects-3K | Ultralytics | AGPL-3.0 | 실내 박스 탐지 |
| ABO | Amazon Berkeley Objects | CC BY 4.0 | 재식별 |
| crack-seg | Ultralytics | Public Domain Mark | 균열(도로·벽, 가구 아님) |
| VisA | amazon-science/spot-diff | CC BY 4.0 | 이상 탐지(산업 부품, 가구 아님) |

상세 근거·제외 데이터: `ai/training/ai1/docs/model_and_data_selection.md`, `ai/training/ai1/configs/datasets.yaml`.

## 실행 (GPU 서버, 컨테이너 안)

```bash
cd ai/training/ai1
bash scripts/run_all.sh --profile standard     # smoke 점검 → standard 본 실행, 결과 outputs/*.tar.gz
bash scripts/run_all.sh --profile max          # 정확도 최대화(COCO+LVIS 병합 라벨, 전체 데이터, GPU 2장)
```
서버 실행 담당자용 안내: `ai/training/ai1/HANDOFF.md`. 단계별 재개·오류 대응: `ai/training/ai1/docs/troubleshooting.md`.

## 결과 (2026-11-25, 연구실 GPU 서버 RTX PRO 4500 Blackwell 1장, standard 프로필 — 실측)

| 지표 | B0 (학습 전) | B1 YOLO11n-seg | B1 YOLO11s-seg |
|---|---|---|---|
| 전체 23종 mask mAP50-95 (LVIS 테스트 954장) | 0.143 | 0.339 | **0.371** |
| 공통 9종 (B0도 아는 품목) | 0.364 | 0.366 | 0.397 |
| 신규 14종 (B0가 모르는 품목) | 0.000 | 0.321 | 0.354 |
| Recall (conf≥0.25) / 누락 수 | 0.234 / 2,559 | 0.337 / 2,215 | 0.377 / 2,082 |
| 학습 시간 | — | 31분 | 49분 |

- 참고(Ultralytics 공식, COCO 80종): YOLO11n-seg 0.320, s 0.378, m 0.415. 데이터가 달라 직접 비교는 불가.
- **나빠진 품목**: 의자 0.167 → 0.091. 원인 확인: COCO val2017에서 의자가 있는 572장 중 LVIS 의자 라벨은 87장(15%).
  → `max` 프로필은 공통 품목 라벨을 COCO 원본(빠짐없는 라벨)으로 교체.
- 재식별(ABO 테스트 877개 상품): Recall@1 히스토그램 0.47 / DINOv2 0.41 / **학습 헤드 0.59**.
- 합성 카메라 영상 40편: 재등장 연결로 중복 49 → 20건, 잘못된 병합 3 → 4건, 물품 F1 0.33 → 0.43.
- B3 합성 시나리오 300개: 분실 의심 F1 수량비교 0.48 → 학습 헤드 매칭 0.91. **실제 이사 전후 영상 평가는 미측정.**
- B4: 균열(실제 crack-seg) F1 0.72, VisA PatchCore 이미지 AUROC 0.69~0.98. **실제 가구 손상 성능은 미측정.**
- 입력 영상(개발용 방 영상)의 물품 목록 정확도는 사람이 정답을 작성해야 측정 가능(검수표 제공).

## 레포에 넣지 않은 것
- 학습된 가중치(.pt), 공개 데이터, 실행 결과 압축, 개인 방 영상·프레임 → Release·공유 드라이브·서버에만 보관

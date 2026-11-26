# AI 1 → AI 2 전달 문서: 물품 인식·추적 결과 사용법

AI 1(물품 탐지·세그멘테이션·추적·재등장 연결)이 지금 무엇을 줄 수 있고, AI 2(크기 측정)가 어떻게 받아 쓰면 되는지
정리한 문서다. 학습·평가 파이프라인 전체 설명은 `docs/ai1-training-pipeline.md`, 코드는 `ai/training/ai1/`에 있다.

## 1. 한 줄 요약

AI 1은 **"영상의 어느 픽셀이 어떤 물건인지(마스크)"와 "여러 프레임에서 같은 물건끼리 묶은 번호"**를 준다.
길이·부피는 재지 않는다. AI 2는 이 마스크 영역을 깊이 지도에 겹쳐 물건별로 크기를 재고, 같은 번호의 여러 프레임을 모아 평균·검증할 수 있다.

## 2. AI 1이 주는 것 (영상 1편 기준)

`infer.py` 실행 폴더(예: `outputs/<run>/video/B1/B1/`) 안의 파일:

| 파일 | 내용 | AI 2 활용 |
|---|---|---|
| `frames.jsonl` | 프레임마다 한 줄: `frame_idx`, `time_sec`, 검출마다 `class_name`(모델 클래스), `category`(짐싹 품목 코드), `conf`, `box_xyxy`, `track_id`(없으면 null), `mask_area_px`, `mask_polygons`(외곽선 다각형 목록) | 프레임별 물체 마스크 → 깊이 지도에서 그 영역만 사용 |
| `items.csv` | 물품 후보(트랙) 목록: `pred_item_id`, `status`(confirmed / needs_review), `category`, 처음·마지막 시각, 대표 프레임, 대표 사진 경로 | 측정 대상 물품 목록 |
| `relink_*/items.csv` | 재등장 연결 후 목록. 여러 트랙이 한 물품으로 묶이면 `G00001`, `track_ids`에 구성 트랙 | 같은 물건이 화면 밖에 나갔다 돌아와도 한 물품으로 측정 |
| `item_embeddings_*.npz` | 물품별 외형 임베딩(재식별용) | 이사 전후 같은 물건 매칭에 사용 가능 |
| `run_manifest.json` | 영상 SHA-256, FPS, 해상도, 회전, 모델 가중치 SHA-256, 설정 | 같은 원본으로 실험했는지 확인 |
| `api/items_B2_final.json` (파이프라인 결과) | 물품별 요약 JSON (아래 3절) | 백엔드·측정 연동 초안 |

**좌표계**: 회전 메타데이터를 적용한 **표시 방향의 원본 해상도 픽셀**(언니방영상 기준 1080×1920), 좌상단 원점, xyxy.
리사이즈·레터박스 좌표가 아니다. `mask_polygons`는 외곽선을 1px 오차로 단순화했고 구멍은 표현하지 않는다.
**`track_id`는 한 영상 안에서만 유효한 임시 번호**다(`docs/ai-contract.md`의 정의와 같음).

## 3. 물품 요약 JSON (`api/items_*.json`, schema `jimssak.item.v1`)

물품마다 `item_id, tracking_id, linked_tracking_ids, class_name, confidence, first_seen_timestamp, last_seen_timestamp,
representative_image, bounding_box{frame_idx, xyxy}, segmentation_mask{polygons}, reid_embedding_reference,
matched_item_id, matching_confidence, missing_status, damage_status, ..., dimensions_cm(null), box_fit_60x40x40("unknown"),
review_required, review_reason`.

- `dimensions_cm`, `volume_liter`는 **AI 1이 채우지 않는다(null)**. AI 2가 채울 자리다.
- 박스 수납 여부는 치수 없이 판단하지 않으므로 `unknown`(판단 보류).

## 4. 팀 공통 규격과의 관계 (아직 연결 안 됨)

현재 서비스 분석기(`ai/service/worker.py`)는 인식 단계를 `--recognizer 패키지.모듈:함수`로 받는다.
이 함수는 영상 경로를 받아 `ai.service.pipeline.Recognition`(안에 `VideoInput.views`)을 돌려줘야 한다.
**AI 1 출력 → `Recognition` 변환 어댑터는 아직 만들지 않았다.** 필드 대응은 다음과 같고, 회전·해상도 필드는 AI 2와 확인이 필요하다.

| AI 1 (`frames.jsonl`) | `ObjectView` (`ai/measurement/moving_contract.py`) | 비고 |
|---|---|---|
| `frame_idx` | `frame_index` | 원본 프레임 번호(0부터) |
| `track_id` (int) | `track_id` (str) | 재등장 연결 후에는 그룹 ID로 바꿔 넣을지 결정 필요 |
| `category` | `label` | 짐싹 품목 코드(sofa, bed, desk, ...) |
| `box_xyxy` | `bbox` | 원본 px |
| `mask_polygons` 중 가장 큰 조각 | `mask_polygon_px` | 여러 조각일 때 처리 규칙 합의 필요 |
| (영상 메타) | `annotation_size_px`, `pixel_rotation_ccw` | AI 1 좌표는 '회전 적용 후' 기준 → AI 2 쪽 정의와 맞춰야 함 |
| 트랙별 평균 conf | `Recognition.confidences[track_id]` | |

## 5. 지금 성능 (2026-11-25 서버 standard 실행, 실측)

| 항목 | 결과 | 데이터 |
|---|---|---|
| 물품 세그멘테이션(23종) mask mAP50-95 | 학습 전 0.14 → 학습 후 0.37 (YOLO11s-seg) | LVIS 독립 테스트 954장 (공개 데이터) |
| 마스크 품질: 맞게 찾은 물체의 윤곽 IoU 평균 | 0.83 | 같은 테스트 |
| 찾은 비율(Recall, conf≥0.25) | 0.38 | 같은 테스트 |
| 중복 등록(같은 물건 여러 번) | 49 → 20건 (재등장 연결) | 합성 카메라 영상 40편 |

**AI 2가 알아야 할 한계**
- 의자 성능이 학습 후 떨어짐(0.17 → 0.09). 원인: LVIS 라벨에서 의자의 85%가 빠져 있음(확인함). COCO 라벨을 합친 개선 실험(`max` 프로필)을 다시 돌릴 예정.
- 언니방영상 육안 관찰(정답 아님): 붙박이장 문짝이 수납장 8~9개로 쪼개짐, 사무용 의자 누락. 실제 영상 정확도는 정답 작성 전이라 미측정.
- 마스크는 '보이는 부분'만이라 가려진 물건은 실제보다 작게 측정될 수 있다.

## 6. 직접 돌려 보기

```bash
cd ai/training/ai1
python -m pip install -r requirements.txt      # torch는 기존 환경 것을 사용
python -u infer.py --video <영상.mp4> --video-id <id> --label B1 \
    --weights <B1_yolo11s-seg_best.pt> --device cuda:0 --output-root outputs/ai2_test
python -u link_reappearance.py --run outputs/ai2_test/B1/<run 이름>
```

- 학습된 가중치는 Git에 넣지 않고 Release로 공유한다:
  [ai1-weights-standard-20261125](https://github.com/OSS-team-A/jimssak/releases/tag/ai1-weights-standard-20261125)
  (`B1_yolo11s-seg_best.pt` 19.6MB 권장, `SHA256SUMS.txt`로 무결성 확인).

  ```bash
  gh release download ai1-weights-standard-20261125 -R OSS-team-A/jimssak \
      -p "B1_yolo11s-seg_best.pt" -p "SHA256SUMS.txt" -D weights
  ```
- 개인 방 영상·프레임은 레포에 올리지 않는다.

## 7. 다음에 맞춰야 할 것
1. AI 1 → `Recognition` 어댑터(`--recognizer ai...:recognize`) 구현과 회전·해상도 규칙 합의
2. 재등장 연결 그룹 ID를 측정 단위로 쓸지(트랙 단위 vs 물품 단위) 결정
3. 같은 원본 영상(SHA-256 기록)으로 AI 1 → AI 2를 끝까지 돌려 실측 오차표 작성

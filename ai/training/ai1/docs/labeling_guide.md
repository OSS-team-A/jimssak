# 정답·검수·라벨링 가이드 (사람이 하는 일)

모델 출력이나 ChatGPT 답변을 정답으로 복사하지 않는다. 정답은 사람이 원본 영상을 보고 실물 기준으로 작성한다.

---

## 1. `annotations/ground_truth_items.csv` — 영상별 실제 물품 목록

**모델 결과를 보기 전에** 원본 영상만 보고 작성한다. Excel에서 열고, 저장할 때 **CSV UTF-8(쉼표로 분리)** 형식을 고른다.

| 열 | 작성법 | 예시 |
|---|---|---|
| `video_id` | `infer.py --video-id`와 같은 값 | `unnie_room_01` |
| `item_id` | 영상 안에서 실물 하나당 고유 ID. 이름이 비슷해도 다른 실물이면 다른 ID | `bed_01`, `chair_01`, `chair_02` |
| `category` | `configs/taxonomy.yaml`의 code 중 하나 (없으면 `other`) | `bed` |
| `box_fit` | `fits` / `no_fit` / `unknown` — 60×40×40cm 박스에 현재 형태 그대로 넣고 닫을 수 있는가 | `no_fit` |
| `box_fit_basis` | 판정 근거. 실측값, AI2 결과, 또는 판단 불가 이유 | `실측 200x150x30cm` |
| `rep_time_sec` | 그 실물이 가장 잘 보이는 시각(초) | `3.5` |
| `rep_frame_idx` | (선택) 해당 프레임 번호 = 시각 × FPS(약 29.97) | `105` |
| `reappear_times_sec` | 화면에서 사라졌다 다시 나오는 시각들(`;` 구분) | `20.8` |
| `labeler` | 작성자 |  |
| `note` | 비고. `other`면 실제 이름, 검수 중 추가한 행이면 `검수 중 추가` |  |

규칙
- **대상(no_fit) 후보만이 아니라** 영상에 보이는 가구·가전 등 운반 판단이 필요한 실물을 모두 적고 box_fit을 매긴다.
  (박스에 들어가는 물건도 `fits`로 적어 두어야, 모델이 그것을 찾았을 때 '오검출'이 아니라 '범위 외'로 처리된다.)
- 치수를 모르면 `unknown`. 추측으로 `no_fit`을 만들지 않는다. `unknown` 물품은 별도 표로 보고된다.
- 바닥의 매트리스·이불 더미처럼 품목이 애매하면 실물 기준으로 정하고 `note`에 이유를 적는다.

## 2. `annotations/match_review_<run>.csv` — 예측 ↔ 정답 연결

```
python make_review_sheet.py --items runs/B0/<run>/items.csv --out annotations/match_review_B0_<run>.csv
```

- 행마다 `rep_images`(대표 사진, `;` 구분)와 `items_contact_sheet.jpg`, 결과 영상(`annotated.mp4`)의 해당 시각을 보고:
  - `gt_item_id`: 그 예측이 가리키는 실물의 `item_id`. 실재하지 않는 물체면 비운다.
  - `extra_gt_item_ids`: 한 예측이 서로 다른 실물 여러 개를 덮고 있으면 나머지 실물 ID (`;` 구분).
  - `error_type`: `미지원 품목`/`탐지 누락`/`클래스 혼동`/`마스크 오류`/`추적 끊김`/`재등장 중복`/`잘못된 병합`/`오검출`/`기타`
- **놓친 정답**: `pred_item_id`를 비운 새 행을 추가하고 `gt_item_id`와 `error_type`(누락 원인)을 적는다.
- 같은 실물을 가리키는 예측이 여러 개면 모두 같은 `gt_item_id`를 적는다(채점에서 첫 번째만 TP, 나머지는 중복 FP).
- `status=needs_review` 행도 연결한다(자동 집계에는 안 들어가지만 '확인 요청 중 실제 대상 수'를 센다).
- B2(재등장 연결) 결과는 B0 트랙 검수표를 재사용하므로 따로 검수하지 않아도 된다. B1은 트랙이 달라지므로 새로 만든다.

## 3. CVAT 분할 라벨링 (파인튜닝용)

1. **프레임 추출** (환경 창): `data_tools/make_split.py` → `data_tools/extract_frames.py` (README 참고).
   다른 방·각도·조명·가림이 포함되도록 여러 영상에서 뽑는다.
2. **CVAT 프로젝트 만들기**: 라벨 이름을 `configs/taxonomy.yaml`의 **code(영문)** 그대로 만든다
   (`bed`, `mattress`, `sofa`, `chair`, `table`, `desk`, `wardrobe`, ...). 도구는 **Polygon**을 쓴다.
3. **task 만들기**: `frames_v1/train`, `frames_v1/val`, `frames_v1/test` 폴더의 이미지를 업로드(분할별 task 권장).
   이미지 파일명을 바꾸지 않는다(분할을 파일명으로 찾는다).
4. **라벨링 규칙**
   - 사진 안의 **학습 대상 품목은 모두** 라벨링한다. 일부만 고르면 나머지가 '배경'으로 학습되어 성능이 떨어진다.
   - 물품 한 개 = 다각형 하나. **보이는 윤곽**만 따라 그린다(가려진 부분을 상상해서 그리지 않음).
     한 물체가 가림으로 두 조각이면 같은 물체로 묶는다(CVAT 그룹 또는 한 개 shape의 여러 조각).
   - 이불·옷이 덮인 침대처럼 경계가 애매하면 **실물(가구)의 보이는 영역**을 기준으로 일관되게 그린다.
   - 대상 물품이 하나도 없는 사진은 그대로 두고 변환 시 `--empty-as-negative`로 포함 여부를 정한다.
5. **내보내기**: task(또는 project) → Export dataset → **COCO 1.0** 형식. 압축을 풀면 `annotations/instances_default.json`.
6. **변환·검수** (환경 창):
   ```
   python data_tools/cvat_coco_to_yolo.py --coco <export>/annotations/instances_default.json \
       --frames-root /workspace/jimssak_data/frames_v1 --out /workspace/jimssak_data/datasets/v1
   python data_tools/check_labels.py --data /workspace/jimssak_data/datasets/v1/data.yaml \
       --frames-manifest /workspace/jimssak_data/frames_v1/frames_manifest.csv --samples 20
   ```
   `label_check/` 오버레이를 열어 클래스·윤곽·누락을 눈으로 확인한다.
7. **첫 소량 라벨(예: 방 2~3개, 50~100장)** 은 파이프라인 점검용이다. 짧게 학습(`--epochs 10`)해 끝까지 돌아가는지 확인한 뒤 본 라벨링을 늘린다.

## 4. 독립 평가용으로 추가 촬영할 장면

최종 test는 **언니방과 다른 방·다른 실물**이어야 한다. 촬영 규칙(모든 영상 공통):
- 세로 또는 가로 중 하나로 통일, 20~60초, 천천히 한 바퀴 + 처음 위치로 돌아오기(재등장 장면 확보).
- 방마다 촬영 세션 1~2편. 같은 방의 영상은 같은 분할에만 넣는다.

권장 장면 (각 항목이 test에 최소 1회 이상 등장하도록):

| 목적 | 장면 |
|---|---|
| 미지원 품목 학습·평가 | 책상, 옷장, 서랍장, 책장, 수납장·협탁, 화장대, 모니터, 세탁기, 행거, 전신거울, 매트리스 단독, 선풍기 |
| 클래스 혼동 | 모니터와 TV가 같이 있는 방, 책상과 식탁, 소파형 침대·모듈 소파, 바닥 매트리스 |
| 재등장 중복 | 같은 가구를 지나쳤다가 다시 비추기(10초 이상 간격), 방 한 바퀴 돌아 시작 지점 복귀 |
| 잘못된 병합 | **같은 종류가 여러 개**(의자 2~4개, 같은 수납장 2개)가 나란히 있는 장면 |
| 가림·조명 | 옷·이불로 덮인 의자·침대, 어두운 방, 역광 창가, 좁은 방에서 가구 일부만 보이는 장면 |
| 박스 경계 사례 | 60×40×40cm 근처 크기 물건(소형 협탁, 전자레인지, 소형 선풍기, 캐리어) — **실측값 기록** |

각 실물의 치수(가로×세로×높이 cm)를 줄자로 재어 `box_fit_basis`에 적으면 박스 필터 평가까지 할 수 있다.

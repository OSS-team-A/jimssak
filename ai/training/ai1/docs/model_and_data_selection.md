# 공개 데이터셋·모델 선정 근거

조사일 2026-11-25. 아래 '확인' 표시는 개발 PC에서 실제 HTTP 요청·파일 열기로 확인한 것이다. 서버에서 다시 받은 파일의
크기·무결성·라이선스 파일은 실행 결과의 `datasets/integrity.csv`, `datasets/licenses.csv`, `datasets/downloads.jsonl`에 남는다.

## 1. 선택한 데이터셋

| 데이터셋 | 용도 | 다운로드(확인한 크기) | 라이선스(확인 출처) | 라벨 형식 | 규모(확인) | 한계 |
|---|---|---|---|---|---|---|
| **LVIS v1** | 이사 물품 인스턴스 세그멘테이션 학습·평가 (B1) | 주석 zip: train 350.3MB, val 64.0MB (`dl.fbaipublicfiles.com`) + COCO 이미지 개별 다운로드 | 주석: LVIS 홈페이지 기준 CC BY 4.0(홈페이지가 JS라 자동 확인 실패 → 수동 확인 필요로 표기). 이미지: json `licenses` 필드의 Flickr 이미지별 라이선스(CC BY-NC-SA 2.0 등) 확인 | COCO JSON 다각형 마스크, 연합(federated) 주석 | val json: 이미지 19,809(그중 COCO val2017 출신 4,809), 주석 244,707, 카테고리 1,203 | 연합 주석이라 이미지별로 일부 카테고리만 정답(학습 라벨 누락 잡음). 실내 전용 데이터가 아님 |
| **COCO 2017 이미지** | LVIS 이미지 원본 | 이미지별 URL(예: val2017/000000000139.jpg 161,811B 확인) | 이미지별 Flickr 라이선스 | - | - | - |
| **HomeObjects-3K** | 실내 가구 객체 탐지(박스) 학습, 실내 도메인 독립 테스트, 옷장(wardrobe) 보완 | 409.4MB zip | AGPL-3.0 (Ultralytics 문서 dataset license 필드) | YOLO 박스(마스크 없음) | 2,689장(train 2,285 / val 404), 12클래스 | 촬영 공간 정보 없음 → train 안 검증 분리가 같은 방을 공유할 수 있음 |
| **crack-seg** | 균열 손상 세그멘테이션 학습·평가 (B4) | 96.1MB zip | Public Domain Mark 1.0 (Ultralytics 문서) | YOLO 다각형 | 4,029장(train 3,717 / val 200 / test 112). train은 원본 1,238장의 증강본(확인), 분할 간 원본 중복 0(확인) | 도로·벽 균열 — 가구 손상과 재질·형태 다름 |
| **ABO** | 재식별(같은 상품 다중 시점) 학습·평가, 전후 매칭 합성 평가, 합성 손상 기반 이미지 | listings 87.5MB, images.csv.gz 6.4MB, 이미지 개별(최대 256px) | CC BY 4.0 (홈페이지 + tar 안 LICENSE-CC-BY-4.0.txt 확인) | 상품 메타데이터 JSON | listing 147,702개. 가구 관련: CHAIR 2,100 / SOFA 1,199 / TABLE 936 / LAMP 544 / SUITCASE 429 / OTTOMAN 403 / STOOL_SEATING 393 등(확인) | 카탈로그 사진(흰 배경·연출·치수 도면 섞임). 실제 방 영상과 조명·가림·흐림이 다름 |
| **VisA** | 비정형 표면 이상(PatchCore) 학습·평가 (B4) | 1,929.8MB tar | CC BY 4.0 (spot-diff README 확인) | 이미지+이상 마스크, `split_csv/1cls.csv` | 12범주. 범주별 학습(정상) 450~905장, 테스트 정상 50~101 + 이상 100장(확인) | 산업 부품(PCB·캡슐·양초 등) — 가구 손상과 동일하지 않음 |

## 2. 제외한 데이터셋

| 데이터셋 | 제외 이유 |
|---|---|
| MVTec AD | 다운로드 페이지가 신청 양식 뒤에 있어 무인 자동화 불가, CC BY-NC-SA 4.0 → VisA로 대체 |
| Objects365 | 박스만(마스크 없음), 전체 수백 GB → 쿼터·시간 대비 비효율 |
| Open Images V7 | 마스크가 큰 묶음 zip 단위, fiftyone 추가 의존. full 확장 후보로만 기록 |
| ADE20K | Ultralytics 설정은 의미 분할용, 원본은 수동 다운로드 안내 |
| COCO 원본 주석 | 같은 이미지에서 LVIS가 가구 카테고리를 더 많이 제공 → LVIS 사용(COCO 가중치는 B0로 사용) |
| 가구 파손 데이터(Roboflow Universe 등) | API 키·계정 필요 → 무인 자동화 불가. **실제 가구 손상 성능은 미측정** |
| CO3D, TAO 등 다중 시점·추적 | 수 TB 또는 하위 데이터별 별도 동의 → 합성 추적 벤치와 ABO 재식별로 대체 |

## 3. 클래스 확정 방식

후보(요청 목록)를 `configs/taxonomy.yaml`에서 LVIS 카테고리에 **같은 개념만** 매핑했다. 실제로 데이터가 있는지는 전처리에서
학습·테스트 인스턴스 수로 자동 판정하고(`min_train_instances`, `min_test_instances`), 근거를 `datasets/class_availability.csv`에 남긴다.
- LVIS val에서 확인한 인스턴스 예: chair 2,368 / cabinet 1,617 / suitcase 1,791 / box 1,389 / lamp 854 / mirror 573 / monitor 598 /
  sofa 528 / bed 435 / refrigerator 406 / television_set 397 / desk 360 / microwave_oven 191 / fan 156 / air_conditioner 146 /
  dishwasher 74 / mattress 74 / dresser 39 / bookcase 17 / automatic_washer 10 / wardrobe 0 / armoire 1.
- 옷장(wardrobe)은 마스크가 사실상 없어 세그멘테이션 학습 불가 → HomeObjects-3K 박스로 **탐지 전용 모델**에서만 학습.
- dining_chair/office_chair, storage_rack, desktop_computer, floor_lamp는 해당 라벨이 없어 제외·통합(taxonomy의 excluded_candidates).
- 박스 라벨을 마스크로 바꾸지 않는다. 마스크 지표는 마스크 정답이 있는 데이터(LVIS, crack-seg, VisA)에서만 계산한다.

## 4. 모델·알고리즘 선택

| 기능 | 비교 대상(실제 실행) | 선택 기준 | 비고 |
|---|---|---|---|
| 탐지·세그멘테이션 | B0: COCO 사전학습 yolo11n-seg / B1: LVIS 파인튜닝 yolo11n-seg, yolo11s-seg (full: +m) | **검증(val) mask mAP50-95 최고** 모델을 B1으로 자동 선택, 테스트로 보고. 정확도·지연시간·파라미터·학습시간 함께 비교 | Loss: Ultralytics 기본(box CIoU+DFL, cls BCE, mask BCE), Optimizer auto(SGD/AdamW), cosine LR, patience early stop, mosaic·HSV·flip·scale 증강, 체크포인트·resume, OOM 시 batch 절반 재시도, GPU 2장이면 DDP |
| 실내 박스 탐지 | B0(COCO) vs HomeObjects 파인튜닝 yolo11s(full: m) | 실내 도메인 독립 테스트(HomeObjects val) | 옷장·스탠드 박스 탐지 보완 |
| 추적 | BoT-SORT vs ByteTrack (Ultralytics 기본 설정) | 합성 영상 HOTA·IDF1·IDSW, 물품 F1 | 실제 영상 추적 정답 없음 → 합성 |
| 재등장 연결 | 없음 / 색 히스토그램 규칙(기존) / DINOv2 임베딩 / 학습 헤드 임베딩 | 합성 영상의 중복률·잘못된 병합률·재등장 연결 정확도·물품 F1 | 임계값은 검증용 합성 영상에서 선택 |
| 재식별 | HSV 히스토그램 / DINOv2 ViT-S/14(사전학습, 학습 없음) / DINOv2 + triplet 학습 헤드 | ABO 테스트 상품(그룹 분리)의 Recall@1·mAP·쌍 AUROC | 백본 고정 + 헤드 학습(batch-hard triplet, AdamW, cosine, early stop, resume). 비용이 작아 단일 GPU |
| 전후 매칭(B3) | 수량만 비교(기준선) / 히스토그램·DINOv2·학습 헤드 + 코사인 + Hungarian | 분실 의심 F1·정상 오탐률·매칭 정확도 | 임계값은 val 시나리오에서 선택, 경계 구간은 uncertain(사람 확인) |
| 손상 탐지(B4) | 균열: COCO(학습 전, crack 미지원) vs crack-seg 파인튜닝 YOLO-seg / 이상 탐지: 전역 kNN(기준선) vs PatchCore | crack-seg 테스트 mask mAP·IoU·Dice·P/R/F1, VisA 이미지·픽셀 AUROC | PatchCore는 경사하강 학습이 아니라 정상 패치 메모리 구축 |
| 전후 상태 변화(B4) | 픽셀 차이(기준선) vs 정합 + 특징 차이(물품별 PatchCore 방식) | 합성 쌍의 신규 손상 F1·AUROC·영역 IoU·조명/시점 변화 오탐률 | **합성 결과로 실제 가구 손상 성능을 주장하지 않음** |

### 검토했지만 구현하지 않은 것
- **EfficientAD**: 학생-교사 네트워크 학습과 ImageNet 보조 데이터(penalty) 준비가 필요해 서버 1회 실행 시간 안에 안정적으로 넣기 어렵다고 판단. PatchCore로 이상 탐지 기준을 세우고, 필요하면 같은 평가 틀(`damage_stage.patchcore_eval`)에 추가할 수 있게 둠.
- **DINOv2 백본 전체 미세조정**: ABO 학습 상품 수(standard 약 2,800개) 대비 과적합 위험과 비용이 커 백본은 고정하고 투영 헤드만 학습.
- **YOLO 기반 가구 손상 탐지**: 가구 손상 라벨 공개 데이터를 인증 없이 확보하지 못함 → 균열(crack-seg)로 YOLO 손상 세그멘테이션 학습 경로만 검증.

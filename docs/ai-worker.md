# AI 분석기 (백엔드 연결)

백엔드가 영상 분석 작업을 만들면, AI 분석기가 가져가서 분석하고 v2.0 결과를 돌려준다.
결과 형식은 `backend/app/schemas/analysis.py`의 `AnalysisResultV2`다.

## 흐름

1. `POST /api/v1/internal/analyses/claim`: 대기 작업을 하나 가져간다. 없으면 204를 받고,
   5초 뒤에 다시 확인한다.
2. `GET .../{analysis_id}/video`: 영상을 임시 폴더에 내려받는다. 분석이 끝나면 지운다.
3. 가구 인식: 프레임마다 가구 종류, 마스크, 신뢰도를 구한다. AI 1 담당 단계다.
4. 크기 측정: Depth Anything V2를 쓴다(맥에서 프레임당 약 0.06초).
5. 크기 보정 → 침대 표준 규격 맞추기 → 확인 필요 여부 판단
6. `POST .../{analysis_id}/result`: v2.0 결과를 보낸다.

진행률은 영상을 받은 뒤 0.05, 인식이 끝나면 0.2, 측정이 끝나면 0.9로 보고한다.

## 실행

백엔드는 `ANALYSIS_WORKER_KEY`를 같은 값으로 설정한 채 띄운다.

```bash
ANALYSIS_WORKER_KEY=비밀키 python -m ai.service.worker --server http://localhost:8000 --recognizer 패키지.모듈:함수
```

| 옵션 | 뜻 |
|---|---|
| `--recognizer 패키지.모듈:함수` | 가구 인식 함수. 영상 경로를 받아 `ai.service.pipeline.Recognition`을 돌려준다 |
| `--prepared-job job.json` | 인식 단계 대신, 측정 작업 파일에 미리 적힌 관찰을 그대로 쓴다(시험용) |
| `--once` | 작업 하나만 처리하고 끝낸다 |
| `--region kr` / `us` | 침대 규격(기본값 한국) |
| `--device auto` / `mps` / `cpu` / `cuda` | 깊이 모델을 돌릴 장치 |

## 인식 단계 연결 (AI 1)

`Recognition`은 다음 두 가지를 담는다.

- `video`: `VideoInput`. 프레임별 `views`(`frame_index`, `track_id`, `label`, 마스크)와
  카메라 정보가 들어 있다.
- `confidences`: `track_id`별 인식 신뢰도(0~1).
  - 비어 있으면 1.0을 넣고 `confidence_source: "not_provided"`를 붙인다.

가구가 하나도 없으면 `ai.service.pipeline.NoFurniture`를 던진다.

## 결과에 덧붙이는 필드

백엔드는 모르는 필드도 저장한다.

| 필드 | 내용 |
|---|---|
| `track_id` | 인식 단계의 물체 ID |
| `review_codes` | 확인 필요 이유 코드. 예: `few_views`, `missing_depth` |
| `standard_size` | 규격으로 맞춘 침대의 규격. 예: `{"code": "queen", "label_ko": "퀸"}` |
| `snapshots[].frame_index` | 사진으로 쓸 프레임 번호 |

`snapshots[].image_url`은 `null`이다. AI가 사진을 올리는 API가 아직 없어서, 백엔드가
`time_ms` 시각의 화면을 영상에서 잘라 쓰는 방식을 제안했다.

## 실패 코드

모두 `retryable: true`다. 다시 촬영하면 해결될 수 있다.

| 코드 | 언제 |
|---|---|
| `no_furniture_found` | 가구를 찾지 못함 |
| `recognition_failed` | 인식 단계 오류 |
| `measurement_failed` | 크기 측정 오류(카메라 추적 실패 등) |

## 확인 결과

개발 장면 영상 하나(47430485, 13.6초, 가구 3개)를 mp4로 바꿔 끝까지 돌려 봤다.

1. 로컬 백엔드에 올렸다.
2. `--prepared-job`으로 분석기를 돌렸다. CPU에서 36초 걸렸다.
3. 백엔드에 완료 결과가 저장됐다.

| 가구 | 추정(가로×깊이×높이) | 실제 |
|---|---|---|
| 의자 | 48 × 41 × 73 | 43 × 38 × 76 |
| 테이블 | 78 × 55 × 83 | 73 × 46 × 78 |
| 테이블 | 58 × 47 × 64 | 43 × 33 × 44 |

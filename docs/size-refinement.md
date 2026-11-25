# 크기 추정 후처리: 치수별 모델 결합과 확인 필요 판단

`ai.measurement.refine`은 `ai.measurement.moving_video` 결과를 받아 두 가지를 한다.
모델 가중치 없이 실행된다.

1. **치수별 모델 결합(`fuse`)**: 같은 영상을 여러 깊이 모델로 측정한 결과에서
   치수마다 다른 모델의 값을 고른다.
2. **확인 필요 판단(`review`)**: 사용자가 확인해야 할 물품과 이유를 만든다.
   v2 응답의 `needs_review`, `review_reasons`에 그대로 쓴다.

## 치수별 모델 결합

ARKit 카메라 위치가 있는 영상은 가로를 Depth Pro, 깊이·높이를 Depth Anything V2에서
가져온다. 선택한 모델이 값을 못 구하면 다른 모델 값을 쓰고, 각 치수의
`depth_model_family`에 출처를 남긴다. 모델 간 값 차이가 중앙값의 25%를 넘으면
`depth_models_disagree_<축>` 표시를 붙인다.

카메라 위치가 없는 일반 영상은 아직 근거가 부족해 기존처럼 Depth Pro만 쓴다.

ARKitScenes 튜닝 장면 6개, 치수 18개의 평균 절대오차(cm):

| 방식 | 가로 | 깊이 | 높이 | 전체 |
|---|---:|---:|---:|---:|
| Depth Anything V2 (기존 기본값) | 11.40 | 6.84 | 6.85 | 8.36 |
| Depth Pro | 6.34 | 11.70 | 12.17 | 10.07 |
| 치수별 결합 | 6.34 | 6.84 | 6.85 | 6.68 |
| 치수별 결합, 장면 하나씩 빼고 고른 경우 | 6.34 | 8.70 | 6.85 | 7.30 |

결합 규칙을 같은 6개 장면에서 골랐으므로 6.68cm는 실제보다 좋게 나온 값이다.
각 장면을 빼고 나머지 5개로 모델을 골라 그 장면을 채점하면 7.30cm다. 가로는 모든 경우에
Depth Pro가 뽑혔다. 유리 테이블(42899679)을 빼도 결합이 단일 모델보다 낫다
(5.67 → 4.97cm, 장면을 빼고 고르면 5.24cm).
**개발용 새 장면으로 다시 확인하기 전까지는 잠정 규칙이다.**

```bash
python -m ai.measurement.moving_video job.json --model depth-anything-v2 --output outputs/dav2.json
python -m ai.measurement.moving_video job.json --model depth-pro --output outputs/dp.json
python -m ai.measurement.refine fuse \
  --result depth-anything-v2=outputs/dav2.json --result depth-pro=outputs/dp.json \
  --output outputs/fused.json
PYTHONPATH=. python experiments/size-estimation/compare_depth_models.py truth.json \
  outputs/dav2.json outputs/dp.json outputs/fused.json --output outputs/fusion-comparison.json
```

## 확인 필요 판단

| 코드 | 조건 | 사용자에게 보이는 이유 (예) |
|---|---|---|
| `unavailable` | 크기를 하나도 못 구함 | 크기를 추정하지 못했어요. 직접 입력해 주세요 |
| `missing_<축>` | 해당 치수만 못 구함 | 옆면이 충분히 촬영되지 않아 깊이를 구하지 못했어요 |
| `few_views` | 프레임별 중앙값을 썼거나 시점 3개 미만 | 여러 각도에서 촬영되지 않아 크기가 부정확할 수 있어요 |
| `approximate_outline` | 마스크 없이 사각형만 있음 | 윤곽 대신 사각형으로 측정해 크기가 부정확할 수 있어요 |
| `models_disagree` | 모델 간 차이 25% 초과 | 추정 방식마다 값 차이가 커서 확인이 필요해요 |
| `out_of_range_<축>` | 품목별 일반 크기 범위 밖 | 옷장 높이 311cm는 일반적인 크기(120~270cm)를 벗어나요 |

값은 바꾸지 않고 표시만 한다. 품목별 범위는 개발용 ARKitScenes 장면 361개의 가구
3,420개 실제 크기에서 1~99백분위를 20% 넓혀 정했다. 실제 크기 중 1.0%만 범위를 벗어난다.
책상·옷장·서랍장은 ARKitScenes에 없는 품목이라 직접 정했다.

```bash
python -m ai.measurement.refine review outputs/fused.json --output outputs/review.json
# 개발용 가구 목록이 늘면 범위를 다시 계산
python -m ai.measurement.refine ranges outputs/arkitscenes-dev-objects.csv
```

기존 결과 16건(ARKit/일반 영상, 옷장, GLEN 책장)에 적용하면, 표시된 9건의 치수별 최대
상대오차 중앙값은 57%, 표시되지 않은 7건은 15%였다. 놓친 큰 오차는 일반 영상의
GLEN 책장(111%)이다. 크기는 범위 안이고 여러 시점이 결합돼 지금 기준으로는 잡히지 않는다.

## 남은 검증

- 개발용 장면 20~30개를 더 받아 두 모델로 측정하고 결합 규칙을 다시 확인한다. 순서는 [개발용 장면 평가](dev-evaluation.md)에 있다.
- 일반 영상(카메라 위치 없음)에서도 치수별 결합이 나은지 같은 방식으로 비교한다.
- 최종 평가용 장면은 규칙을 확정한 뒤 한 번만 돌린다.

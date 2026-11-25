# ARKitScenes 개발용·최종 평가용 분할

지금까지 크기 추정 개선은 같은 ARKitScenes 장면 몇 개로 튜닝과 평가를 함께 했다.
이 분할은 한 번도 보지 않은 장면을 최종 평가용으로 떼어 두기 위한 것이다.

## 규칙

- 공식 Validation 폴드에서 3D 주석이 있는 장면만 대상이다.
- 같은 방(`visit_id`)의 영상은 항상 같은 쪽에 둔다. 같은 방의 다른 영상이 개발용에
  있으면 가구와 배치가 같아서 평가가 새는 것을 막는다.
- 기존 실험(`outputs/`)과 문서에 한 번이라도 등장한 장면이 있는 방은 모두 개발용이다.
- 나머지 방은 `visit_id` 해시로 40%를 최종 평가용으로 정한다. 다시 실행해도 같다.

결과는 [arkitscenes-split.json](../experiments/size-estimation/arkitscenes-split.json)에
고정했다. `outputs/`는 git에 올라가지 않으므로 이 JSON이 팀 공용 기준이다.

| 항목 | 수 |
|---|---:|
| Validation 방 / 장면 | 183 / 549 |
| 기존 실험에 쓴 장면 | 155 |
| 기존 실험 때문에 개발용으로 고정된 방 | 40 |
| 최종 평가용 방 / 장면 | 58 / 169 |

## 사용법

최종 평가용 장면은 튜닝, 모델 선택, 오류 분석에 쓰지 않는다. 개선을 마치고
결과를 보고할 때 한 번만 돌린다.

```bash
# 개발용 작업에 최종 평가용 장면이 섞였는지 검사 (섞이면 종료 코드 1)
python -m ai.training.arkitscenes_split check \
  experiments/size-estimation/arkitscenes-split.json outputs/job.json

# 최종 평가 작업이 최종 평가용 장면만 쓰는지 검사
python -m ai.training.arkitscenes_split check \
  experiments/size-estimation/arkitscenes-split.json outputs/final-job.json --final

# 최종 평가용 가구 목록과 정답 치수(cm). Apple 서버에서 3D 주석만 받는다.
python -m ai.training.arkitscenes_split objects \
  experiments/size-estimation/arkitscenes-split.json --side holdout \
  --annotations outputs/arkitscenes-annotations --download \
  --output outputs/arkitscenes-holdout-objects.csv

# 개발용 가구 목록 (새 개발 장면을 고를 때)
python -m ai.training.arkitscenes_split objects \
  experiments/size-estimation/arkitscenes-split.json --side development \
  --metadata outputs/arkit-depth-fix/metadata.csv \
  --annotations outputs/arkitscenes-annotations --download \
  --output outputs/arkitscenes-dev-objects.csv
```

정답 치수는 3D 주석의 중력 정렬 박스에서 바닥 평면의 긴 변을 가로, 짧은 변을 깊이,
수직 축을 높이로 계산한다. 기존 6개 평가 가구의 `truth.json`과 소수점 둘째 자리까지
일치한다. 15cm 미만이나 3.5m 초과, 기울어진 박스는 제외한다.

분할 파일은 덮어쓰지 않는다. 다시 만들어야 하면 `--force`를 쓰고 이유를 커밋
메시지에 남긴다.

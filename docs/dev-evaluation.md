# 개발용 장면으로 크기 추정 평가하기

튜닝에 쓰던 ARKitScenes 장면 6개 대신, 한 번도 쓰지 않은 개발용 장면 수십 개로
크기 추정을 평가한다. 최종 평가용 장면([분할 기준](arkitscenes-split.md))은 쓰지 않는다.

## 만드는 것

`ai.training.arkitscenes_jobs`가 장면마다 가구 상자가 화면에 다 들어오는 구간을 찾아,
iPhone 촬영 앱과 같은 형식(`capture.avi`, `camera.json`, `observations.json`)으로 다시 만든다.
그다음 `ai.measurement.arkit_capture`로 같은 프레임의 영상 두 개를 만든다.

- `<장면>-<가구>`: 기록된 ARKit 카메라 위치를 쓰는 영상
- `<장면>-<가구>-rgb`: 카메라 위치 없이 일반 영상처럼 추정하는 영상

마스크를 단 프레임(기본 6장) 사이의 프레임도 0.2초 간격으로 넣는다. 예전 실험 영상은
마스크 프레임만 1초 가까이 떨어져 있어서, 일반 영상의 카메라 추적이 끊기고 프레임마다
따로 계산되는 일이 많았다(8건 중 3건). 실제 휴대폰 영상은 프레임이 이어져 있으므로 이 방식이
더 실제에 가깝다.

**마스크는 3D 정답 상자를 화면에 투영한 볼록 다각형이다.** 탐지 모델의 마스크 오차를
빼고 크기 추정만 평가하기 위해서다. 상자 윤곽이라 실제 가구 윤곽보다 조금 크다. 예를 들어
테이블 다리 사이 빈 공간까지 포함한다. 같은 프레임의 YOLO 마스크는 99%가 이 다각형
안에 들어가고, 둘의 IoU는 0.6 정도였다. 카메라 위치 변환은 예전 실험의 `camera.json`과
값이 같은 것을 확인했다.

## 맥에서 실행하는 순서

저장소 루트에서 가상환경을 켠 뒤 한 줄씩 실행한다. 1번은 Apple 서버에서 내려받으므로
인터넷이 필요하다.

1. 개발용 장면 25개로 측정 작업 만들기 (장면당 가구 최대 2개)

```bash
python -m ai.training.arkitscenes_jobs experiments/size-estimation/arkitscenes-split.json --metadata outputs/arkit-depth-fix/metadata.csv --scenes 25 --cache outputs/arkitscenes-cache --output outputs/dev-eval
```

2. Depth Anything V2로 측정

```bash
python -m ai.measurement.moving_video outputs/dev-eval/job.json --model depth-anything-v2 --device auto --output outputs/dev-eval/dav2.json
```

3. Depth Pro로 측정

```bash
python -m ai.measurement.moving_video outputs/dev-eval/job.json --model depth-pro --device auto --output outputs/dev-eval/dp.json
```

4. Depth Pro, 카메라 추적 기록 6프레임으로 측정 (일반 영상 시점 결합 개선 확인용)

```bash
python -m ai.measurement.moving_video outputs/dev-eval/job.json --model depth-pro --device auto --track-history 6 --output outputs/dev-eval/dp-h6.json
```

5. 치수별 결합

```bash
python -m ai.measurement.refine fuse --result depth-anything-v2=outputs/dev-eval/dav2.json --result depth-pro=outputs/dev-eval/dp.json --output outputs/dev-eval/fused.json
```

6. 채점

```bash
PYTHONPATH=. python experiments/size-estimation/evaluate_dev.py outputs/dev-eval/truth.json dav2=outputs/dev-eval/dav2.json dp=outputs/dev-eval/dp.json dp-h6=outputs/dev-eval/dp-h6.json fused=outputs/dev-eval/fused.json --output outputs/dev-eval/evaluation.json
```

6번은 ARKit 영상과 일반 영상을 따로 표로 보여 준다. 열은 치수별 평균 절대오차(cm),
상대오차 10% 이내 비율, 측정한 치수 비율(`cover`), 여러 시점 결합으로 구한 비율(`multi`)이다.
모든 실행이 공통으로 측정한 치수만 평균하므로, 어려운 물품을 건너뛴 실행이 더 좋아 보이지 않는다.
`best by axis`는 이 개발용 장면에서 치수별로 가장 좋은 실행이다. 결합 규칙은 이 결과로 정한다.

## 시간

2~4번은 일반 영상 프레임이 많아서 오래 걸린다. Depth Anything V2는 맥 GPU에서 프레임당
약 0.1초라 수십 분 이내지만, Depth Pro는 훨씬 느려서 한 시간 이상 걸릴 수 있다. 먼저
`--scenes 5`로 전체 흐름을 확인한 뒤 25개로 늘리는 것을 권한다.

## 확인할 것

- 일반 영상의 `multi`가 예전 6개 장면(61%)보다 올라가는지
- `dp-h6`가 `dp`보다 일반 영상 오차가 낮은지. 낮으면 기본값을 6으로 바꾼다
- ARKit 영상에서 `fused`가 `dav2`보다 낮은지. 튜닝 장면 6개에서는 8.36 → 6.68cm였다

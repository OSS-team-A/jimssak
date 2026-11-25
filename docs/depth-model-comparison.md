# 가구 크기 추정용 깊이 모델 비교

기존 크기 추정 파이프라인에 세 모델을 연결했다. 각 모델의 출력은 원본 프레임 픽셀에
맞춘 미터 단위 깊이 배열이다.

- [Depth Anything V2 Metric Indoor Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf)
- [Apple Depth Pro](https://huggingface.co/apple/DepthPro-hf)
- [ZoeDepth NYU](https://huggingface.co/Intel/zoedepth-nyu)

Depth Pro는 공식 후처리에서 모델의 시야각으로 배율을 적용하고 역깊이를 미터 단위로
변환한다. ZoeDepth는 좌우 반전 예측을 평균하고 입력 패딩을 제거한다. 모델마다 같은
영상의 같은 프레임, 마스크, 카메라 보정값을 쓴다. 가구 정답 치수는 추정이 끝난 뒤
채점에 사용한다.

## 평가 입력

[ARKitScenes](https://machinelearning.apple.com/research/arkitscenes)의 실제 촬영
여섯 건을 사용했다. 평가 대상은 나무 테이블(48458732), 수납장(47332886),
침대 옆 협탁(47334115), 접이식 테이블(47333934), 원목 협탁(42898538),
유리 테이블(42899679)이다. ARKit 자세를 제공한 18개 치수의 평균 절대오차를
선택 기준으로 삼고, ARKit 자세 없이 측정한 `IMG_3898.MOV`는 별도로 기록한다.

## 재현

`.[vision,depth]` 의존성과 각 공식 모델의 `config.json`,
`preprocessor_config.json`, `model.safetensors`가 필요하다.
가중치와 원본 영상은 로컬에서 읽는다. 영상 입력과 실측치 파일은 서로 분리한다.
파일 형식은 [촬영 연동](arkit-capture.md)과 [채점 예제](examples/moving-img3898-truth.json)를 참고한다.

```bash
python -m pip install -e '.[vision,depth]'

python -m ai.measurement.moving_video /path/to/job.json \
  --model /path/to/depth-anything-checkpoint --device auto --output outputs/dav2.json
python -m ai.measurement.moving_video /path/to/job.json \
  --model /path/to/depth-pro-checkpoint --device auto --output outputs/depth-pro.json
python -m ai.measurement.moving_video /path/to/job.json \
  --model /path/to/zoedepth-checkpoint --device auto --output outputs/zoedepth.json

PYTHONPATH=. python experiments/size-estimation/compare_depth_models.py \
  /path/to/truth.json outputs/dav2.json outputs/depth-pro.json outputs/zoedepth.json \
  --output outputs/depth-comparison.json
```

비교 스크립트는 모든 모델이 값을 반환한 같은 ARKit 치수로 평균 절대오차를 계산하고,
채점 가능한 전체 치수 수와 모델별 산출 수를 함께 기록한다.

## 결과와 기본 모델

모든 모델이 ARKit 치수 18개를 반환했다. 표는 Mac M4에서 동일한 입력과 측정 코드로
비교한 치수 절대오차의 평균이다.

| 모델 | 가로 MAE (cm) | 깊이 MAE (cm) | 높이 MAE (cm) | 전체 MAE (cm) |
|---|---:|---:|---:|---:|
| Depth Anything V2 | 11.40 | 6.84 | 6.85 | 8.36 |
| Depth Pro | 6.34 | 11.70 | 12.17 | 10.07 |
| ZoeDepth | 15.57 | 8.17 | 10.70 | 11.48 |

ARKit 자세가 있는 영상은 전체 MAE가 가장 낮았던 **Depth Anything V2 Metric Indoor Small**을
자동 선택한다.
Depth Pro는 가로 오차가 가장 낮았다. ARKit 자세가 없는 `IMG_3898.MOV`의 상판
가로·깊이·높이 평균 오차는 Depth Pro 11.12cm, Depth Anything V2 38.63cm,
ZoeDepth 45.14cm였다. 해당 영상의 전체 폭은 평가 가능한 결과가 없었다.

`--model`을 생략하면 영상별로 선택한다. `capture_poses`가 있으면 Depth Anything V2,
없으면 **Depth Pro**를 사용한다. 결과의 `depth_model_family`에서 선택을 확인할 수 있다.
`--model depth-pro`, `--model zoedepth`, `--model depth-anything-v2`를 지정하면
해당 모델을 작업의 모든 영상에 사용한다.
모델 이름으로 선택하면 저장소의 `outputs/depth-checkpoints/<모델 이름>`을 먼저
찾고, 그 폴더에 가중치와 설정 파일이 없으면 로컬 Hugging Face 캐시에서 찾는다.
직접 설치한 경로를 넘기는 방식도 지원한다.

사용한 가중치 SHA-256

- Depth Anything V2 `e990eb82fbf11b05b7813261196a2b841bdcf5a05f64396724a8987fa90504a3`
- Depth Pro `9c6811e3165485b9a94a204329860cb333a79877e757eb795a179a4ea34bbcf7`
- ZoeDepth `b616e347efc30e64822be28d2464c2fcd0b665b5fdbc9a135d53d168c31c77ab`

## 새 영상에서의 자동 선택 검증

두 RGB 입력에서 카메라 위치 자료를 사용하지 않고 같은 프레임·마스크·보정 조건으로
기존 Depth Anything V2와 자동 선택된 Depth Pro를 비교했다. 기준 치수는 채점에만 썼다.

| 영상 | 가로 오차 (cm) | 깊이 오차 (cm) | 높이 오차 (cm) |
|---|---:|---:|---:|
| 새 옷장 장면 41254246 | 68.3 → 2.3 | 76.4 → 25.0 | 123.6 → 1.8 |
| 기존 0.5배 방 영상의 GLEN 책장 | 91.3 → 69.0 | 26.3 → 30.9 | 57.4 → 91.2 |

새 옷장 영상은 ARKitScenes 원본에서 RGB 프레임 네 장만 가져와 구성했다. ARKit의
자세와 내부 행렬은 전달하지 않았다. 3D 주석을 기준으로 채점했으며, 앞면이 주로 보여
깊이 오차는 25.0cm 남았다. 0.5배 일반 영상은 왼쪽 책장 한 대의 제품표
`82 × 29 × 138.8cm`를 기준으로 채점했다. 렌즈 초점거리와 위쪽 방향은 영상에서
근사해 두 모델에 동일하게 제공했다. 이 영상에서는 Depth Pro가 여러 시점을 3D로
결합하지 못해 프레임별 추정값의 중앙값을 반환했다.

ARKit 테이블 영상과 일반 방 영상을 한 요청에 넣어 각각 Depth Anything V2와
Depth Pro가 선택되는 것도 확인했다.

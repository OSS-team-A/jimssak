# 가구 크기 추정 학습 실험 3건

가구 폭을 이미 아는 상황에서 높이나 앞뒤 길이를 추정하는 세 가지 실험을 기록한다. 데이터와 평가 단위가 서로 달라 아래 행끼리 성능을 직접 비교하기는 어렵다.

| 실험 | 학습 자료와 평가 단위 | 기준 방식 → 학습 방식 |
|---|---|---|
| [합성 영상 높이 보정](synthetic-height/observed-results.json) | 합성 장면 24,000개 학습, 별도 합성 3,000개 평가. 실영상 평가는 같은 책장 모델의 3프레임 | 합성 높이 평균 절대 백분율 오차 **5.26% → 4.56%**, 실영상 제품표 대비 높이 평균 절대 오차 **3.69 → 2.99 cm** |
| [ARKitScenes 깊이·높이 회귀](arkitscenes/observed-results.json) | 7개 스캔·47개 물체로 학습, 별도 2개 스캔·16개 물체 평가 | 종류별 중앙값 대비 깊이 대용값 평균 절대 오차 **15.63 → 9.12 cm**, 높이 **35.51 → 18.36 cm** |
| [유튜브 조립 영상 길이 보정](youtube-calibration/observed-results.json) | 5편·12프레임·제품 4종. 각 제품을 평가할 때 그 제품의 모든 영상을 학습에서 제외 | 기하 계산 대비 깊이 평균 절대 오차 **2.03 → 2.92 cm**, 높이 **5.25 → 5.96 cm** |

합성 실험은 앞면의 수동 모서리와 근사 초점거리를 받는다. ARKitScenes 실험은 정답 3D 상자를 투영한 가구 영역과 가로로 긴 축을 받는다. 가로로 짧은 축을 깊이 대용값으로 평가하므로 실제 가구 정면의 앞뒤 방향과 다를 수 있다. 유튜브 실험은 수동으로 표시한 상판 모서리와 다리 끝을 받으며, 제품표를 비교 기준으로 사용한다. 세 실험에서 모서리 자동 추출이나 폭 기준 없는 절대 치수는 검증하지 않았다.

유튜브 실험의 보정 모델은 분리 평가에서 두 축의 오차가 증가했다. 측정 기능에는 연결하지 않았다. [LACK 흰색](https://www.youtube.com/watch?v=ynasFcWWafg&t=92s) · [LACK 검정](https://www.youtube.com/watch?v=DwKD3YwDZrw&t=60s) · [KNARREVIK](https://www.youtube.com/watch?v=1WOHBAcVeOQ&t=236s) · [NESNA](https://www.youtube.com/watch?v=0kexzFkS9bE&t=1883s) · [MALM](https://www.youtube.com/watch?v=H_Se6AsZDIE&t=766s)

## 재현

저장소 루트에서 Python 3.12 환경에 NumPy와 OpenCV를 설치한다. 합성 높이 보정에만 PyTorch가 필요하다.

```sh
python -m pip install 'numpy>=2,<3' 'opencv-python-headless>=4.10,<6'
python -m pip install 'torch>=2.5,<3'
python experiments/size-estimation/synthetic-height/train_length.py
python experiments/size-estimation/youtube-calibration/train.py

python experiments/size-estimation/arkitscenes/collect.py
python experiments/size-estimation/arkitscenes/extract.py
python experiments/size-estimation/arkitscenes/train_depth.py
```

ARKitScenes는 Apple 서버에서 선택한 스캔의 카메라·주석과 프레임을 내려받는다. 합성 실험의 실영상 평가를 다시 실행하려면 `--real-corners`로 로컬 주석 JSON을 전달한다. 각 행의 키는 `corners`(1080 × 1920 이미지의 왼쪽 위, 오른쪽 위, 오른쪽 아래, 왼쪽 아래), `reference_width_cm`, `catalog_height_cm`, `focal_hint_px`다. 실영상의 원본과 비공개 좌표, 모델 바이너리는 로컬에 보관한다. [Apple ARKitScenes 데이터 안내](https://github.com/apple-aiml-research/ARKitScenes/blob/main/DATA.md)

`observed-results.json`은 최초 실행 결과다. 재실행은 각 폴더의 로컬 결과 파일을 갱신한다.

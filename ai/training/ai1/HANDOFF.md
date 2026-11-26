# 짐싹 AI — GPU 서버 실행 안내서 (실행 담당자용)

이 문서는 **GPU 서버 계정을 가진 실행 담당자**를 위한 것입니다. 코드 수정, 데이터 찾기, 라벨링은 필요 없습니다.
아래 순서대로 파일을 올리고, **명령 한 줄**을 실행한 뒤, 생긴 결과 파일 하나를 돌려주시면 됩니다.

> **두 번째 실행(정확도 개선판 `max`)을 부탁받으셨다면 맨 아래 "8. 개선판(max) 실행"만 보시면 됩니다.**

## 0. 하시는 일 (전부)

1. 받은 압축 파일을 서버에 올리기
2. 압축 풀기
3. 전용 실행 환경 만들기(또는 고르기)
4. 실행 명령 한 줄 입력
5. 끝나면 결과 압축 파일 내려받기
6. 그 파일을 전달하기

## 1. 미리 알아 두실 것 (자동화할 수 없는 조건)

| 항목 | 내용 |
|---|---|
| **인터넷** | 서버(컨테이너)에서 아래 주소로 HTTPS/HTTP 다운로드가 가능해야 합니다. 계정·API 키·로그인은 **필요 없습니다**.<br>`pypi.org`(파이썬 패키지), `github.com`(YOLO 가중치·데이터), `dl.fbaipublicfiles.com`(LVIS 주석), `images.cocodataset.org`(COCO 이미지), `amazon-berkeley-objects.s3.amazonaws.com`(ABO), `amazon-visual-anomaly.s3.us-west-2.amazonaws.com`(VisA), `huggingface.co`(DINOv2), `download.pytorch.org`(WideResNet) |
| **라이선스** | 동의 클릭 절차는 없습니다. 공개 데이터 중 일부 이미지(COCO/Flickr)는 이미지마다 비상업 조건(CC BY-NC 등)이 붙어 있고, YOLO는 AGPL-3.0입니다. **연구·교육(공개SW프로젝트) 목적 사용**을 전제로 하며, 이 판단은 프로젝트 담당자(코드 작성자) 책임입니다. 상세: 결과 압축 안 `datasets/licenses.csv` |
| **디스크** | standard: 여유 공간 **40GiB 이상** 권장(실사용 추정 약 10~15GiB, 아래 표). 쿼터에 데이터·캐시가 포함됩니다 |
| **GPU** | 1장으로 동작합니다. 컨테이너에 GPU가 2장 보이면 탐지 모델 학습을 자동으로 2장(DDP)에 나눠 씁니다. GPU가 하나도 안 보이면 **CPU로 바꾸지 않고 멈춥니다** |
| **시간** | smoke 점검이 먼저 돌고(수십 분 이내로 예상), 이어서 standard 본 실행이 수 시간 걸립니다. 정확한 시간은 서버에서 아직 재 본 적이 없어 **미측정**이며, smoke가 끝나면 로그에 `학습 시간 추정` 줄이 출력됩니다 |
| **서버 설정** | sudo, apt, 드라이버, Docker 설정은 건드리지 않습니다. 패키지는 프로젝트 폴더 안 전용 가상환경(`.venv`)에만 설치되고, 기존 PyTorch는 그대로 재사용합니다(바뀌면 스스로 중단) |

## 2. [내 PC] 서버로 올리기

`<계정>`, `<서버주소>`는 본인 접속 정보입니다.

```powershell
scp jimssak_project.tar.gz <계정>@<서버주소>:~/workspace/
```

## 3. [서버 창] 압축 풀기·환경 준비

```bash
cd ~/workspace && tar -xzf jimssak_project.tar.gz
lab-status
lab-env list
```
- `lab-status`에서 GPU를 다른 사람이 쓰고 있지 않은지 확인해 주세요.
- 이 작업 전용 환경을 권장합니다(평소 쓰시는 환경의 패키지에 영향을 주지 않도록). 없으면 하나 만듭니다.
  ```bash
  lab-env create jimssak 0
  ```
  마지막 숫자는 사용할 GPU 번호(0 또는 1)입니다. GPU 두 장을 함께 쓰는 환경을 만드는 방법은 서버 사용 가이드를 따라 주세요(한 장이어도 정상 동작).
  기존 환경을 쓰신다면 아래 명령의 `jimssak`을 그 환경 이름으로 바꾸세요.

## 4. [서버 창] 실행 — 이 한 줄

```bash
lab-env run jimssak "cd /workspace/jimssak && bash scripts/run_all.sh --profile standard"
```

자동으로 진행되는 일: 전용 가상환경·패키지 설치 → GPU 확인 → **smoke 점검(전 과정 축소 실행)** → standard 본 실행
(사전학습 모델 다운로드 → 학습 전 기준선 → 공개 데이터 다운로드·검사·전처리 → 모델 학습 → 평가 → 보고서 → 결과 압축).

진행 확인:
```bash
lab-env jobs jimssak
ls -t ~/workspace/jimssak/logs/ | head -3
tail -n 30 ~/workspace/jimssak/logs/<위에서 본 가장 최근 run_all_... 파일>
```
`~/workspace/logs/`(lab-env 자체 로그)에도 같은 출력이 남습니다.

끝났다는 표시: 로그 마지막에
```
==== 전체 완료 ... ====
돌려줄 파일: ... jimssak_results_standard_<시각>.tar.gz
```

## 5. [내 PC] 결과 받아서 전달

```powershell
scp "<계정>@<서버주소>:~/workspace/jimssak/outputs/jimssak_results_standard_*.tar.gz" .
```
이 파일(standard 결과) 하나를 전달해 주세요. 같은 폴더의 `jimssak_results_smoke_*.tar.gz`(점검 결과)도 함께 주시면 도움이 됩니다.

## 6. 문제가 생겼을 때

**고치려 하지 마시고** 다음 두 가지를 보내 주세요: ① `~/workspace/jimssak/logs/` 의 가장 최근 `run_all_*.log`
② `~/workspace/jimssak/outputs/` 안 가장 최근 폴더의 `stage_status.csv`

| 로그에 보이는 것 | 의미 | 하실 일 |
|---|---|---|
| `CUDA GPU가 보이지 않습니다` | 컨테이너에 GPU 미연결 | 환경 GPU 설정 확인 후 같은 명령 재실행 |
| `설치 후 torch가 바뀌었습니다` | 패키지 충돌 | 로그 전달 |
| `다운로드 실패` / `다운로드 재시도` 반복 | 인터넷·방화벽 | 1번 표의 주소 접속 가능 여부 확인 후 재개 명령 |
| `디스크 여유 ... < 예상 필요량` | 쿼터 부족 | 공간 확보 후 재개 명령 |
| `[단계명] 실패` | 해당 단계 오류 | 로그 전달 (또는 아래 재개 명령 한 번 시도) |

**이어서 하기(재개)** — 성공한 단계는 건너뛰고 실패한 단계부터 다시 합니다:
```bash
lab-env run jimssak "cd /workspace/jimssak && bash scripts/run_all.sh --profile standard --resume latest"
```

**중간에 멈추기**: `lab-env enter jimssak` → `ps -ef | grep run_pipeline` 으로 PID 확인 → `kill <PID>`.
환경 전체를 중지하면 그 안의 다른 작업도 함께 종료됩니다. 멈춘 뒤에는 위 재개 명령으로 이어서 할 수 있습니다.

## 7. 다 끝난 뒤 정리(선택)

결과 파일을 전달한 뒤 `~/workspace/jimssak` 폴더(데이터·모델 포함)를 지우셔도 됩니다. 디스크 쿼터에 포함됩니다.
Hugging Face·PyTorch 캐시는 `~/workspace/.cache/` 아래에 있습니다.

## 참고: 프로필별 규모

| 프로필 | 용도 | 탐지 모델 | 데이터(LVIS 학습/검증/테스트 이미지 상한) | 추정 디스크 사용 |
|---|---|---|---|---|
| smoke | 자동 점검 | yolo11n-seg 2 epoch, 320px | 160 / 40 / 60 | 약 1GiB |
| **standard** (기본) | 발표용 | yolo11n-seg·yolo11s-seg 각 60 epoch, 640px + 실내 탐지 yolo11s 60 epoch + 균열 yolo11n-seg 50 epoch | 12,000 / 1,200 / 2,000 | 약 10~15GiB |
| full | 확장 | n·s·m-seg 각 100 epoch + 실내 탐지 yolo11m | 40,000 / 3,000 / 5,000 | 약 20~30GiB |

디스크 추정 근거: 다운로드 파일 크기(실측: LVIS 주석 0.41GB, VisA 1.93GB, HomeObjects 0.41GB, crack-seg 0.10GB, ABO 0.09GB)와
COCO 이미지 평균 약 160KB × 이미지 수, 압축 해제·학습 결과 포함. 실행 시간은 서버 미측정(smoke 후 자동 추정치 확인).

## 8. 개선판(max) 실행 — 두 번째 실행

첫 실행(standard)은 끝났습니다(감사합니다). 이번에는 정확도를 높인 `max` 설정으로 한 번 더 부탁드립니다.

**바뀐 점(담당자분이 하실 일은 같습니다)**: 학습 데이터가 몇 배로 늘고, 더 큰 모델을 학습하며, **GPU 2장**을 함께 씁니다.

| 항목 | 내용 |
|---|---|
| GPU | **두 장 모두 보이는 환경**이면 자동으로 2장을 나눠 씁니다. 한 장만 보이면 한 장으로 돌고 더 오래 걸립니다. 두 장을 함께 쓰는 환경을 만드는 방법은 서버 사용 가이드(`/usr/local/share/pilab/AI_GUIDE.md`)를 따라 주세요. 연구실 규칙상 GPU 두 장을 오래 쓰게 되니 `lab-status`로 비어 있는지 확인하고 필요하면 관리자와 시간을 조율해 주세요 |
| 시간(추정) | 약 10~14시간 (첫 실행의 실측 속도로 계산한 추정치. 실제 값은 로그의 `학습 시간 추정` 줄) |
| 디스크 | 여유 **80GiB 이상** (COCO 이미지 묶음 19.3GB를 한 번에 받아 필요한 사진만 꺼냅니다) |
| 추가 다운로드 주소 | 첫 실행과 같습니다(`images.cocodataset.org`에서 큰 zip 파일을 받음) |

**순서**
1. [내 PC] 새로 받은 `jimssak_project.tar.gz`를 서버에 올립니다(첫 실행 때와 같은 명령).
2. [서버 창] 같은 자리에 압축을 풉니다. 코드 파일만 새것으로 바뀌고, 이미 받은 데이터·가상환경(`data/`, `.venv/`)은 그대로 재사용됩니다.
   ```bash
   cd ~/workspace && tar -xzf jimssak_project.tar.gz
   lab-status
   lab-env list
   ```
3. [서버 창] 실행 (환경 이름은 실제 사용하는 것으로):
   ```bash
   lab-env run jimssak "cd /workspace/jimssak && bash scripts/run_all.sh --profile max"
   ```
   smoke 점검(수 분)이 먼저 돌고, 통과하면 max 본 실행이 이어집니다.
4. 끝나면 `~/workspace/jimssak/outputs/jimssak_results_max_<시각>.tar.gz`를 전달해 주세요
   (`jimssak_results_smoke_…`도 함께 주시면 좋습니다).

중간에 멈추면 같은 명령 끝에 `--resume latest`를 붙여 이어서 실행합니다:
```bash
lab-env run jimssak "cd /workspace/jimssak && bash scripts/run_all.sh --profile max --resume latest"
```
오류가 나면 고치지 마시고 `~/workspace/jimssak/logs/`의 가장 최근 `run_all_max_….log`와
`outputs/` 가장 최근 폴더의 `stage_status.csv`를 보내 주세요.

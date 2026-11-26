# 오류 대응 안내

모든 실행은 `outputs/<run>/stage_status.csv`와 `state.json`에 단계별 상태를 남긴다. 실패하면 **그 단계에서 멈추며 성공으로 기록하지 않는다.**
원인을 해결한 뒤 같은 run을 이어서 실행한다:

```bash
bash scripts/run_all.sh --profile standard --resume latest          # 실패 단계부터
bash scripts/run_all.sh --profile standard --resume <run 이름> --from-stage train_detection   # 특정 단계부터 다시
```

| 증상(로그) | 원인 | 조치 |
|---|---|---|
| `CUDA GPU가 보이지 않습니다` | 컨테이너에 GPU가 연결되지 않음 | `lab-env list`에서 환경 GPU 확인, 필요 시 새 환경 생성. CPU로 돌리려면 `--allow-cpu`(점검용, 발표 결과로 쓰지 않음) |
| `설치 후 torch가 바뀌었습니다` | 의존성이 다른 torch를 요구 | 로그의 pip 출력 확인. `.venv`를 지우고 재실행(`rm -rf .venv`) — 기본 이미지 torch는 영향 없음 |
| `cv2 import 실패(libGL)` | GUI용 OpenCV | 스크립트가 자동으로 headless로 교체. 계속 실패하면 로그 전달 |
| `다운로드 재시도` 반복 / `다운로드 실패` | 네트워크·방화벽·일시 장애 | HANDOFF 1번 표의 주소 접속 확인 후 `--resume latest`(이미 받은 파일은 크기 확인 후 건너뜀, 끊긴 파일은 이어받음) |
| `이미지 N장 다운로드 실패 → 제외` | COCO/ABO 개별 이미지 일부 실패 | 경고만. 제외 수가 `datasets/dataset_stats.json`에 기록됨 |
| `학습 가능한 클래스가 하나도 없습니다` | 데이터가 거의 안 받아짐 | 다운로드 로그 확인 후 재개 |
| `분할 누수 발견` | 데이터 분할 오류 | 실행 중단(정상). `datasets/split_report.json` 전달 |
| `학습 실패(OOM) → batch N로 새로 학습` | GPU 메모리 부족 | 자동으로 batch를 절반으로 줄여 재시도(최대 `oom_retry`회). 그래도 실패하면 프로필의 batch를 줄여 재개 |
| `YOLO 학습 실패` (DDP) | 다중 GPU 통신 오류 | 자동으로 batch 축소 재시도. 계속되면 GPU 1장 환경에서 재개 |
| `디스크 여유 … < 예상 필요량` | 쿼터 부족 | `data/cache`의 큰 파일(VisA tar 등)은 전처리 후 지워도 됨. 공간 확보 후 재개 |
| 실행 도중 서버 재부팅·컨테이너 재시작 | 학습 프로세스 종료(컨테이너 자동 재시작은 학습을 잇지 않음) | 같은 명령에 `--resume latest`. YOLO는 `last.pt`에서, 재식별 헤드는 `last.pt`에서 이어서 학습 |

## 단계 이름

`env_check, deps_check, validate_inputs, pretrained_download, baseline_b0_video, datasets_download, datasets_verify,
preprocess, split_check, train_detection, train_reid, matching_b3, damage_b4, eval_detection, eval_tracking,
video_reanalysis, compare, report, package`

## 부분 실행 스크립트 (설치 후)
- `bash scripts/download_datasets.sh` — 다운로드·검사만
- `bash scripts/run_baseline.sh` — B0 기준선만
- `bash scripts/run_training.sh` — 전처리·분할·학습만
- `bash scripts/run_evaluation.sh` — 평가·비교·보고서만
- `bash scripts/package_results.sh` — 결과 압축만
(모두 `--profile`, `--resume` 지원. 생략하면 같은 프로필의 가장 최근 실행을 이어서 사용)

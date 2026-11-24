# 📦 짐싹 (Jjimsak)

### 이사 준비를 싹, 짐싹!
**주소와 방 영상만 입력하면 견적 비교부터 계약서 작성까지**

짐싹은 원룸 및 소형 이사를 준비하는 1인 가구를 위한 AI 기반 이사 준비 서비스입니다.

사용자가 주소를 입력하고 방을 촬영하면 AI가 영상 속 물품을 인식하여 물품 목록을 자동으로 생성합니다. 이후 사용자가 확인한 물품 정보와 이사 조건을 바탕으로 업체별 견적을 비교하고 계약서 초안 작성까지 지원합니다.

---

## 1. 프로젝트 소개

### 기획 배경

1인 가구의 이사 준비 과정은 번거롭습니다. 이삿짐이 적더라도 이사 유형을 선택하고, 업체마다 견적을 요청하며, 물품의 종류와 수량 및 크기를 직접 입력해야 합니다.

특히 소규모 이사는 방문 견적 없이 비대면으로 계약하는 경우가 있어 물품 목록이나 작업 조건이 명확하지 않으면 추가 비용이나 분쟁이 발생할 수 있습니다.

짐싹은 이러한 문제를 해결하기 위해 **영상 기반 물품 목록 자동 생성과 이사 조건의 체계적인 관리**를 제공합니다.

### 프로젝트 목표

- 영상 분석을 통한 물품 목록 자동 생성
- 물품의 중복 인식 및 누락 최소화
- 물품 수량, 크기, 예상 무게 정보 제공
- 사용자 확인을 통한 정확한 물품 명세서 작성
- 이사 조건을 반영한 업체별 견적 비교
- 합의사항에 기반한 계약서 초안 작성
- 이사 전후 물품 상태 확인 지원

## 2. 주요 기능

### ① 주소 및 이사 정보 입력

- 출발지 및 도착지 주소 입력
- 이사 날짜 설정
- 층수, 엘리베이터 등 건물·이동 조건 확인
- 추가 작업 조건 및 특이사항 입력

### ② AI 기반 물품 자동 인식

사용자가 촬영한 방 영상을 분석하여 주요 가구와 가전제품을 인식합니다.

- YOLO 기반 객체 탐지 및 세그멘테이션
- 영상 프레임별 물품 종류 식별
- 객체 추적을 통한 동일 물품 추적
- 재등장한 물품의 중복 집계 최소화
- 물품 종류별 수량 자동 집계

### ③ 물품 크기 및 무게 추정

- 영상에서 인식한 물품의 크기 추정
- 깊이 추정 및 영상 처리 기술 활용
- 물품별 예상 크기(cm) 제공
- 물품 종류와 크기에 기반한 예상 무게 정보 제공
- 확인이 필요한 정보 별도 표시

※ 크기 및 무게는 추정값이며 실제 측정값과 차이가 있을 수 있습니다. 정확한 확인이 어려운 정보는 사용자가 직접 수정할 수 있도록 제공합니다.

### ④ 물품 목록 확인 및 수정

AI가 생성한 물품 목록을 사용자가 직접 검토하고 확정합니다.

- 인식된 물품 목록 확인
- 물품별 수량 증가 및 감소
- 누락 물품 직접 추가
- 물품 크기 및 예상 무게 수정
- 분해·조립 필요 여부 확인
- 확인이 필요한 물품 별도 표시

### ⑤ 이사 견적 및 유형 비교

확정된 물품 목록과 이사 조건을 바탕으로 견적을 비교합니다.

- 일반 이사, 반포장 이사, 포장 이사 등 유형 비교
- 업체별 예상 견적 비교
- 포장 및 운반 작업 범위 확인
- 분해·조립 및 추가 비용 조건 비교

### ⑥ 계약서 초안 작성

- 선택한 업체 및 이사 일정 반영
- 물품 명세와 작업 범위 정리
- 견적 금액 및 추가 비용 조건 기록
- 계약서 초안 PDF 생성
- 사용자와 업체의 최종 확인 지원

### ⑦ 이사 전후 물품 상태 확인

- 이사 후 물품 영상 촬영
- 기존 물품 목록과 비교
- 파손 의심 물품 확인
- 영상에서 확인되지 않은 물품 표시

## 3. 서비스 이용 흐름

```text
① 기본 정보 입력
   └─ 출발지 / 도착지 / 이사 날짜
             ↓
② 방 영상 촬영 및 업로드
             ↓
③ AI 기반 물품 분석
   ├─ 물품 종류 인식
   ├─ 객체 추적 및 중복 제거
   └─ 크기 및 무게 추정
             ↓
④ 물품 목록 확인
   ├─ 수량 수정
   ├─ 누락 물품 추가
   └─ 추정 정보 확인
             ↓
⑤ 이사 조건 확인
   └─ 건물 정보 및 작업 조건
             ↓
⑥ 업체별 견적 및 이사 유형 비교
             ↓
⑦ 계약서 초안 작성
             ↓
⑧ 이사 후 물품 상태 확인
```

## 4. 기술 스택

| 구분 | 기술 |
|---|---|
| Frontend | React, TypeScript |
| Backend | Python, FastAPI |
| AI | PyTorch, YOLO |
| Computer Vision | OpenCV |
| Object Tracking | BoT-SORT |
| Depth Estimation | Depth Anything V2 |
| GPU | NVIDIA GPU, RunPod |
| Collaboration | GitHub, GitHub Issues, Pull Requests |

※ 기술 스택은 개발 과정에서 변경될 수 있습니다.

## 5. 시스템 아키텍처

```text
사용자
  │
  ▼
Frontend (React)
  │
  │ REST API
  ▼
Backend (FastAPI)
  │
  ├── 주소 및 이사 정보 관리
  ├── 영상 업로드 및 분석 요청
  ├── 물품 목록 관리
  ├── 견적 비교
  └── 계약서 초안 생성
  │
  ▼
AI Pipeline (Python)
  │
  ├── 영상 프레임 선별
  ├── YOLO 물품 탐지
  ├── 객체 추적
  ├── 재등장 물품 연결
  ├── 크기 추정
  └── 물품 목록 생성
  │
  ▼
분석 결과 반환
  │
  ▼
사용자 검토 및 수정
```

## 6. 프로젝트 디렉터리 구조

프론트엔드, 백엔드, AI를 하나의 GitHub 저장소에서 관리하는 모노레포 구조를 사용합니다.

```text
jimssak/
│
├── frontend/                 # 프론트엔드
│   ├── src/
│   └── package.json
│
├── backend/                  # FastAPI 백엔드
│   ├── app/
│   │   ├── api/
│   │   ├── schemas/
│   │   ├── services/
│   │   └── main.py
│   └── requirements.txt
│
├── ai/                       # AI 분석 모듈
│   ├── detection/            # 물품 탐지
│   ├── tracking/             # 객체 추적 및 중복 제거
│   ├── measurement/          # 크기 추정
│   ├── training/             # AI 학습 및 평가
│   └── pipeline.py           # AI 파이프라인 통합
│
├── docs/                     # 프로젝트 문서
├── .github/                  # GitHub 협업 설정
├── .gitignore
├── .env.example              # 루트 환경변수 예시
├── pyproject.toml            # 공통 Python 패키지/개발 의존성
├── CONTRIBUTING.md           # 협업과 저장소 관리자 설정 안내
└── README.md
```

## 7. 설치 및 실행 방법

현재 제공 범위는 React 화면 → FastAPI → 고정 AI 데모 응답입니다.
실제 영상 업로드, YOLO/BoT-SORT, Depth Anything V2, 견적/계약 및 전후 비교는 후속 구현입니다.
기획된 주요 기능과 현재 구현 범위를 구분해 주세요.

### 준비

Node.js 24 LTS (`.nvmrc`), Python 3.11 이상을 사용합니다.
`python3 --version`을 먼저 확인하고, 시스템 Python이 오래되었다면
아래 `python3` 대신 설치된 `python3.12` 등 3.11 이상의 실행 명령을 사용하세요.

```bash
git clone https://github.com/OSS-team-A/jimssak.git
cd jimssak
cp .env.example .env
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Windows PowerShell에서는 `Copy-Item .env.example .env`,
`python -m venv .venv`, `.venv\Scripts\Activate.ps1`을 사용합니다.

### 백엔드 실행 (저장소 루트)

```bash
python -m uvicorn backend.app.main:app --reload --port 8000
```

- 상태: http://localhost:8000/health
- API 문서: http://localhost:8000/docs
- 데모: `curl -X POST http://localhost:8000/api/v1/analysis/demo`
- backend 폴더에서 실행할 때: `python -m uvicorn app.main:app --reload`
- backend 폴더 기준 requirements 설치: `python -m pip install -r requirements-dev.txt`

### 프론트엔드 실행 (새 터미널)

```bash
cd frontend
npm ci
npm run dev
```

http://localhost:5173 에서 **데모 분석 실행**을 누릅니다.
API가 실행 중이면 의자 샘플 1개가 나오며 크기는 보정 정보 필요로 표시됩니다.
프론트엔드는 루트 `.env`의 `VITE_API_BASE_URL`을 사용합니다.
환경변수 변경 후 앱을 재시작하세요. VITE_ 변수는 브라우저에 공개됩니다.

### AI와 검증 (저장소 루트, 가상환경 활성화)

```bash
python -m ai.pipeline
ruff check .
pytest -q
cd frontend
npm run check
npm run build
```

AI 핵심 규격/데모는 기본 Python 의존성만 사용합니다.
GPU 모델 패키지와 가중치는 모델 선정 및 라이선스 확인 후 별도로 추가합니다.
Python 직접 의존성은 `pyproject.toml`에서 관리하며 requirements 파일은 이를 참조합니다.
Python 의존성은 호환 범위 지정이며 완전한 lock 파일은 아직 없습니다.
프론트엔드는 커밋한 package-lock.json과 `npm ci`를 사용합니다.

### 공통 규격과 협업

- [크기 추정 첫 실험](docs/measurement-start.md)
- [AI 데이터 규격](docs/ai-contract.md), [샘플 JSON](docs/examples/demo-analysis.json)
- [협업 및 main 보호 설정](CONTRIBUTING.md)
- [최초 Issue 작업 목록](docs/first-tasks.md)
- `.github/ISSUE_TEMPLATE/`: 기능/버그 양식
- `.github/PULL_REQUEST_TEMPLATE.md`: PR 양식
- `.github/workflows/ci.yml`: 프론트 검사/빌드, Python 린트/테스트

## 8. 팀원 및 역할

| 담당 | 주요 역할 |
|---|---|
| Frontend | 서비스 UI, 영상 업로드, 물품 목록 및 견적 화면 |
| Backend | FastAPI, API 설계, 물품 정보 관리, 서비스 로직 |
| AI 1 | YOLO 물품 탐지, 객체 추적, 재등장 물품 연결 |
| AI 2 | 깊이 추정, 크기 측정, 추정 정확도 검증 |

## 9. GitHub 협업 규칙

### 브랜치 전략

`main` 브랜치를 기준으로 기능별 브랜치를 생성합니다.

```text
main
├── feat/frontend
├── feat/backend
├── feat/ai-detection
└── feat/ai-measurement
```

### Commit Convention

| 유형 | 설명 |
|---|---|
| feat | 새로운 기능 추가 |
| fix | 버그 수정 |
| refactor | 코드 리팩터링 |
| docs | 문서 작성 및 수정 |
| test | 테스트 코드 작성 |
| chore | 환경 설정 및 기타 작업 |

예시:

```text
feat: YOLO 기반 물품 탐지 기능 구현
feat: 물품 중복 인식 방지 로직 추가
feat: 영상 업로드 API 구현
fix: 물품 수량 집계 오류 수정
docs: README 프로젝트 구조 수정
```

### Issues 및 Pull Requests

- 개발 작업은 GitHub Issue로 관리합니다.
- 작업별 담당자를 지정합니다.
- 기능 구현 시 별도 브랜치를 생성합니다.
- 작업 완료 후 Pull Request를 작성합니다.
- 코드 리뷰 및 확인 후 `main` 브랜치에 병합합니다.

## 10. 검증 계획

동일한 방 영상과 정답 물품 목록을 사용하여 AI 분석 결과의 정확도를 평가합니다.

### 평가 항목

| 평가 항목 | 평가 방법 |
|---|---|
| 물품 수량 오차 | 실제 수량과 AI 예측 수량 비교 |
| 물품 누락 | 정답 목록 중 인식되지 않은 물품 수 |
| 중복 인식 | 동일 물품이 여러 번 집계된 수 |
| 크기 추정 오차 | 실측 크기와 추정 크기 비교 |
| 사용자 수정 시간 | 최종 물품 목록을 확정하는 데 걸린 시간 |

### 비교 실험

- **비교군 A:** 범용 AI를 통한 물품 목록 생성
- **비교군 B:** 물품 추적 및 재등장 연결 로직을 적용한 짐싹

동일한 영상과 물품 범위에서 수량 오차, 누락·중복 및 사용자 수정 시간을 비교하여 개선 여부를 검증합니다.

## 11. 시연 영상 및 실행 화면

개발 완료 후 다음 자료를 추가할 예정입니다.

- 서비스 메인 화면
- 방 영상 업로드 화면
- AI 물품 인식 결과
- 물품 목록 확인 및 수정 화면
- 업체별 견적 비교 화면
- 계약서 초안 작성 화면
- 서비스 시연 영상

---

### 이사 준비를 싹, 짐싹!

**방 촬영으로 시작하는 간편한 이사 준비**

짐싹은 영상 기반 물품 인식과 사용자 확인을 통해 이사 준비 과정의 정보 누락과 반복적인 작성 부담을 줄이고자 합니다.

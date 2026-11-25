# AI 공통 데이터 규격 v1.0

정의: `ai/contracts.py`. AI 모듈 안에서 쓰는 프레임 단위 규격입니다. 백엔드로 보내는 영상 분석 결과는 v2.0 형식이고, 백엔드는 이 모델을 가져다 쓰지 않고 따로 검증합니다.
예시: `docs/examples/demo-analysis.json`, 생성: `python -m ai.pipeline`.

| 필드 | 의미 |
|---|---|
| schema_version / mode | 규격 버전 / demo 또는 model |
| frame.video_id / frame_id | 원본 영상 ID / 0부터 시작하는 원본 프레임 번호 |
| frame.timestamp_ms | 영상 시작 기준 밀리초 |
| width_px / height_px | 원본 프레임 해상도 |
| bbox | 좌상단 원점, x 오른쪽/y 아래, 원본 px, xyxy |
| track_id | 한 영상 안에서 추적하는 임시 ID. 영상 간 동일성 보장 없음 |
| object_id | 재등장/이사 전후 매칭 결과. 매칭 미구현이면 null |
| label / confidence | 클래스명 / 0~1 탐지 점수 |
| mask_polygon_px | 원본 px 다각형. 없으면 null. RLE 미지원 |
| camera.fx_px / fy_px | 원본 해상도 기준 초점거리(px). 보정 없으면 camera=null |
| depth_m | 물체까지 거리(m), 없으면 null. 상대 깊이 맵 값을 그대로 넣지 않음 |
| dimensions | 폭/높이/두께(cm). 추정 불가하면 null |
| measurement_status | needs_calibration 또는 estimated |

## 첫 실험

동일한 원본 영상/프레임을 사용하고 SHA-256, 프레임 추출 간격, 해상도를 기록합니다.
AI 1은 탐지/추적 JSON을, AI 2는 같은 JSON을 입력받아 보정 및 실측 오차를 검증합니다.
리사이즈/레터박스 좌표는 원본 좌표로 역변환한 뒤 전달합니다.
트랙 수는 장면 재등장 중복 제거 후의 실제 물품 수와 다를 수 있습니다.

현재 DemoDetector/DemoTracker는 고정 의자 샘플과 ID를 반환하며 실제 영상 처리를 하지 않습니다.
Detector/Tracker 인터페이스에 YOLO와 BoT-SORT 어댑터를 연결할 수 있습니다.

크기 예시는 `폭px × 깊이m / fx × 100` 식을 사용합니다.
카메라 보정과 metric depth가 모두 있어야 폭/높이를 반환합니다.
정면에 가까운 물체의 bbox 근사치라 회전/가림/배경 포함으로 오차가 생깁니다.
두께는 이 식으로 구할 수 없으므로 null입니다. 단안 상대 깊이는 실제 미터가 아닙니다.

분실/파손 비교는 후속 작업입니다. 영상에서 안 보이는 물품은 미확인,
외관 차이는 파손 의심으로 표시하고 실제 분실/파손으로 단정하지 않습니다.

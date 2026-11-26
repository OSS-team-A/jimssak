# 범용 ChatGPT 비교 절차 (사람이 채팅창에서 수행, API 호출 없음)

이 저장소는 ChatGPT API를 호출하지 않으며 영상을 외부 서비스로 자동 전송하지 않는다.
비교는 사람이 평소 쓰는 ChatGPT 채팅창에서 직접 하고, 결과를 CSV로 옮긴다.

## 1. 입력 조건 (모든 비교에서 동일하게)
- 같은 원본 영상을 업로드하거나, 영상 업로드가 안 되면 **실제로 제공한 프레임**을 기록한다
  (`data_tools/extract_frames.py`로 뽑은 같은 프레임 세트를 쓰면 재현 가능).
- 프롬프트는 `templates/chatgpt_prompt.txt`를 그대로 쓴다. **품목 예시를 주지 않는다.** 박스 기준(60×40×40cm)은 같게 준다.
- 한 영상당 새 채팅에서 한 번 묻는다(이전 대화의 영향을 받지 않게).

## 2. 기록 — `annotations/chatgpt_runs.csv`
| 열 | 내용 |
|---|---|
| `chatgpt_run_id` | 예: `gpt_unnie_room_01_a` |
| `video_id` | 정답 CSV와 같은 값 |
| `date` | 실행 날짜 |
| `model_display_name` | 채팅창에 표시된 모델 이름 그대로 |
| `interface` | 예: `ChatGPT 웹 채팅창` |
| `input_mode` | `영상 업로드` 또는 `프레임 이미지` |
| `n_frames_provided`, `frame_selection` | 프레임을 줬다면 장수와 선택 방법(예: 1초 간격 22장) |
| `prompt_file` | 실제 사용한 프롬프트 파일 |
| `response_file` | 응답 원문을 그대로 저장한 텍스트 파일 경로 (예: `annotations/chatgpt_raw/gpt_unnie_room_01_a.txt`) |
| `elapsed_time_note` | 사람이 잰 대략적인 응답 시간(선택). 토큰 비용은 기록하지 않는다 |

## 3. 응답 옮겨 적기 — `annotations/chatgpt_items.csv`
- 응답의 물건 한 줄 = 한 행. `raw_name`에 ChatGPT가 쓴 이름 그대로, `category`에 taxonomy code로 정규화한 값
  (이름만 보고 정규화한다. 정답 목록을 보고 맞춰 주지 않는다).
- `box_fit`: ChatGPT가 박스에 안 들어간다고 하면 `no_fit`, 들어간다고 하면 `fits`, '판단 어려움'이면 `unknown`.
- 근거 시각/프레임이 있으면 `evidence_time_sec`/`evidence_frame`, `pred_item_id`는 `C001`처럼 부여, `quantity=1`.
- 응답이 "의자 2개"처럼 **합계만** 주면 한 행에 `quantity=2`, `pred_item_id`·근거는 비운다
  → 수량 오차만 계산되고 F1·중복 원인은 '판정 불가'로 표시된다.
- 근거가 있는 행은 `make_review_sheet.py` 대신 같은 열 구조로 `annotations/match_review_chatgpt.csv`를 만들어 `gt_item_id`를 연결한다.

## 4. 채점
```
python evaluate_list.py --label ChatGPT --pred-format chatgpt --gt annotations/ground_truth_items.csv \
    --pred annotations/chatgpt_items.csv --review annotations/match_review_chatgpt.csv
```
- `list` 모드: ChatGPT가 나열한 물건 전체를 사람이 정한 대상 집합과 비교(목록 능력).
- `box` 모드: ChatGPT가 `no_fit`이라 한 것만 최종 목록으로 보고 비교(박스 판단까지 포함한 전체 흐름).
- AI1은 박스 판정을 하지 않으므로 `box` 모드에서 AI1 행은 '미실행'(AI2·실측 판정이 들어오면 `--box-fit-csv`로 채점).
- 발표표에 입력 방식 차이(ChatGPT: 영상/프레임 N장, AI1: 전체 프레임 처리)를 함께 적는다.

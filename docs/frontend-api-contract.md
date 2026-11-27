# 프론트엔드 API 연동 계약 및 검증

기준: `origin/main`의 `24c6c56` (PR #20, #36, #37, #38, #79 머지 후), [운영 OpenAPI](https://api.kyooo.shop/openapi.json), Figma `[최종] 짐싹 ui - 11/24수정` (`531:8299`). 화면의 CSS, 에셋과 주요 문구는 유지한다. 응답 표시용 변환은 `frontend/src/features/moveInfoAdapter.ts`, `historyAdapter.ts`, `review.ts`에 둔다.

## 화면 흐름과 확인한 계약

| 화면 | Figma 백엔드 주석 | 실제 엔드포인트 및 연결 |
| --- | --- | --- |
| SC-01/02 인증 | 계정에 이사 기록 연결 | `POST /auth/signup`, `POST /auth/login`, `POST /auth/refresh`, `POST /moves/{id}/claim` (Bearer와 X-Guest-Token 동시 전달) |
| SC-03/04/04B/04D 촬영 | SC-04B: 촬영 후 `POST /moves/{id}/media` | `POST /moves` → `POST /moves/{id}/media`; 파일 선택·촬영 후 검토 화면을 거친다. 이사 후 촬영은 `POST /moves/{id}/after-media` |
| SC-06 이사 정보 입력/확인 | `PATCH /moves/{id}`, 확인은 `GET /moves/{id}` | 주소 후보의 `road_address`, `building_name`, `adm_cd`, `mountain`, `lot_main`, `lot_sub`와 사용자 상세주소·층을 PATCH. 성공 후 GET의 `elevator`, `building.status`, 일정 값을 표시 |
| SC-06 주소 검색 | `GET /addresses?keyword=` | 실제 검색 후보를 표시하고 선택한 주소 필드를 PATCH에 전달 |
| SC-06 분석 대기/완료/실패 | `GET /moves/{id}`를 2~3초마다 조회; 완료 후 `GET /moves/{id}/analysis` | 3초 폴링, 완료 시 분석 결과·목록으로 이동. `demo=true`는 실제 물품으로 표시하지 않음 |
| SC-07 채팅/목록 | `GET /moves/{id}/analysis`의 `items` 하나가 물품 한 줄 | 최초 분석 출력은 analysis. 사용자가 수정한 현재 목록은 별도 `GET /moves/{id}/inventory`로 읽고 항목 PATCH/DELETE 및 목록 확정을 사용 |
| SC-13 기록 | `GET /moves` | 전체 페이지를 조회해 실제 기록만 표시. 현재 서버의 `stage`는 `preparing`만 지원 |
| SC-14~16 이사 후 | 전후 비교 결과를 표시하는 화면 | 이사 후 영상 업로드까지만 API 확인. 비교 결과·검토 저장은 표시하지 않음 |

위 경로의 공통 접두사는 `/api/v1`이다.

## 운영 서버 직접 검증 (2026-11-27 KST)

- `GET /health` → 200 `{"status":"ok"}`.
- 인증 없이 `GET /api/v1/moves` → 401 `{"detail":"로그인이 필요합니다."}`.
- 무효한 refresh token으로 `POST /api/v1/auth/refresh` → 401 `{"detail":"리프레시 토큰이 만료되었거나 유효하지 않습니다."}`. 클라이언트는 세션을 비우고 로그인 안내를 표시한다.
- `GET /api/v1/addresses?keyword=서울역` → 200, 후보에 Swagger의 주소 필드 8개와 `total`, `page`, `per_page`가 있음.
- 비회원 `POST /api/v1/moves` → 201과 guest token. 그 이사의 GET → `stage=collecting`, `analysis.status=not_started`, `inventory.status=empty`. 분석 GET은 빈 `items`, 목록 GET은 `items=[]`.
- 실제 주소 검색 후보 두 개와 2026-12-20 09:00으로 PATCH → 200, 두 주소·일정 모두 `filled`. 주소별 승강기 결과가 각각 `unknown`과 `unavailable`로 달라 최종 확인 화면이 응답에 따라 표시되는 것을 Chrome에서 확인.
- 분석 전 물품 추가 `POST /inventory/items` → 409 `{"detail":"영상 분석이 완료된 후 목록을 편집할 수 있습니다."}`. **이 409 응답은 현재 OpenAPI의 해당 operation에 명시되지 않았다.** 프론트는 실제 409를 실패로 처리한다.
- `http://localhost:5173` Origin의 운영 API 요청과 `Authorization`, `X-Guest-Token`이 들어간 PATCH 사전 요청의 CORS 허용 확인.
- Chrome 모바일 320px, 390×844, 430px에서 홈·촬영·이사 정보·분석·목록·기록의 가로 넘침을 확인했다. 320px 기록 헤더의 6px 넘침을 수정하고 세 폭에서 재확인했다. 390×844 촬영 준비와 실제 API 저장 후 이사 정보 확인 화면을 캡처해 기존 스타일과 대조했다.
- Chrome에서 실제 guest 이사 생성, 주소 검색 후보 선택, 층·일정 입력, PATCH, GET 응답 기반 최종 확인을 클릭 흐름으로 확인했다. 브라우저 재방문 시 실제 저장된 이사 정보를 다시 불러왔다.
- 무효한 access/refresh token으로 기록 요청 → 401 → refresh 401 → 세션 제거 → 로그인 안내를 Chrome에서 확인했다. 로그인으로 이동한 뒤 뒤로가기·새로고침에서도 기록 예시가 노출되지 않았다.
- 로그인·토큰 갱신 성공, 실제 분석 완료·물품 변경, 영상 파일 업로드 성공, PDF 생성 성공은 테스트 계정·유효 영상·완료된 분석 데이터가 없어 운영 서버에서 검증하지 못했다.

## 백엔드 지원이 확인되지 않은 흐름과 필요한 계약

다음은 **제안 계약이며 현재 존재하는 API가 아니다.** 응답 필드 이름과 상태는 백엔드 확정 전까지 프론트에 연결하지 않는다.

1. **SC-09 예상 비용:** 이사 ID와 확정된 목록 버전을 기준으로 견적을 요청하거나 조회할 경로, 일반·반포장·포장별 금액 범위·통화·산정 근거·산정 시각·거리/현장 조건 반영 여부가 필요하다. 현재는 금액을 표시하지 않는다.
2. **SC-07 채팅 자연어 수정:** 입력 문장을 서버 항목 ID·변경 내용으로 해석하는 요청과, 적용 전 사용자 확인을 위한 구조화된 제안 응답이 필요하다. 현재 채팅 문구는 브라우저에만 보관하며 목록 변경으로 표시하지 않는다. 직접 추가 API는 있지만 Figma의 최종 화면에 별도의 추가 폼이 없어 자연어를 임의로 항목 생성 요청에 연결하지 않는다.
3. **SC-13A/15D~16 이사 후 비교:** 이사 후 업로드한 영상의 조회 가능한 상태·비교 작업 ID, 상태(대기·진행·완료·실패), 원본 물품 ID별 전후 사진/구간과 의심·미확인 이유, 사용자 확인/미발견 선택 저장 응답, 기록 `stage` 전이가 필요하다. 현재 `after-media`는 업로드만 지원하므로 비교 성공·완료 상태를 만들지 않는다.
4. **PDF 서버 발급:** 현재 별도 PDF API가 없어서 프론트가 확정된 실제 목록과 이사 정보를 이용해 A4 파일을 생성한다. 서버 발급이 요구되면 이사 ID·목록 버전·이사 유형을 입력으로 받고 PDF MIME/파일명 또는 서명 URL을 반환하는 계약이 필요하다.
5. **회원 이름:** SC-02B의 이름 입력란에 대응하는 필드가 `SignupRequest`와 `User`에 없다. 이름을 임의 필드로 보내지 않는다.

비회원 기록 연결 API는 확인했으므로 로그인 시 `POST /moves/{id}/claim` 성공 후에만 guest token을 제거한다. 실패하면 비회원 세션을 남겨 재시도할 수 있게 한다.

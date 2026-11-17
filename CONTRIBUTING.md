# 협업 안내

1. 기능/버그 Issue에 목표, 담당자, 완료 기준을 적습니다.
2. 최신 main에서 `feat/<issue>-<topic>`, `fix/<issue>-<topic>` 브랜치를 만듭니다.
3. 작은 단위로 커밋합니다. 예: `feat: add detector adapter`.
4. PR에 `Closes #번호`, 변경 이유와 검증 결과를 적습니다.
5. CI 통과와 팀원 1명 이상 리뷰 후 병합하고 브랜치를 삭제합니다.

Python: 루트에서 `ruff check .`와 `pytest -q`.
Frontend: frontend에서 `npm ci`, `npm run check`, `npm run build`.

공통 데이터 규격 변경은 AI 1, AI 2, Backend 담당자가 함께 검토합니다.
원본 프레임, 해상도, 좌표계, 단위를 변경할 때는 docs/ai-contract.md와 예시를 갱신합니다.

## 관리자가 GitHub에서 설정할 항목

- Settings → General → Features → Issues 활성화 확인.
- Settings → Collaborators and teams에서 실제 팀원 초대.
- Settings → Rules → Rulesets에서 main 대상 규칙 생성.
- PR 필수, 승인 1명, 최신 변경 후 승인 재요청, 대화 해결 필수.
- 최초 CI 실행 후 필수 검사 `frontend`, `backend-ai` 선택.
- 강제 푸시와 main 삭제 차단. 팀 운영에 맞게 관리자 우회 범위 결정.
- 기본 브랜치가 main인지 확인 후 적용. 요금제와 저장소 공개 범위에 따라 지원이 다릅니다.

템플릿과 CI는 파일로 제공되며, 위 계정/저장소 설정은 자동 적용되지 않습니다.
팀원이 확정되지 않아 CODEOWNERS와 담당자 자동 지정은 추가하지 않았습니다.

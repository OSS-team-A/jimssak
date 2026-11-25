from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["system"], summary="서버 상태 확인",
            description="서버가 떠 있으면 `{\"status\": \"ok\"}`를 반환한다.")
def health():
    return {"status": "ok"}

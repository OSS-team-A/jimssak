from fastapi import APIRouter

from ..schemas.analysis import AnalysisResult
from ..services.analysis import analyze_demo

router = APIRouter()


@router.get("/health", tags=["system"], summary="서버 상태 확인",
            description="서버가 떠 있으면 `{\"status\": \"ok\"}`를 반환한다.")
def health():
    return {"status": "ok"}


@router.post("/api/v1/analysis/demo", response_model=AnalysisResult, tags=["analysis"])
def demo():
    """Synthetic fixture only; this endpoint does not accept or analyze video."""
    return analyze_demo()

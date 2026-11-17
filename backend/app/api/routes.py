from fastapi import APIRouter

from ..schemas.analysis import AnalysisResult
from ..services.analysis import analyze_demo

router = APIRouter()


@router.get("/health", tags=["system"])
def health():
    return {"status": "ok"}


@router.post("/api/v1/analysis/demo", response_model=AnalysisResult, tags=["analysis"])
def demo():
    """Synthetic fixture only; this endpoint does not accept or analyze video."""
    return analyze_demo()

"""AI 1 학습·평가 파이프라인(ai/training/ai1)의 채점·추적 지표 단위 테스트를 CI에서 실행한다.

목록 채점 규칙은 표준 라이브러리만 쓰므로 항상 실행한다.
추적 지표·매칭 테스트는 scipy가 있을 때만 실행한다.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

AI1 = Path(__file__).resolve().parents[1] / "training" / "ai1"
sys.path.insert(0, str(AI1 / "src"))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"ai1_{name}", AI1 / "tests" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TestListEval = _load("test_listeval").TestListEval

if importlib.util.find_spec("scipy") is not None:
    _metrics = _load("test_metrics_modules")
    TestMot = _metrics.TestMot
    TestMatching = _metrics.TestMatching
    TestRetrieval = _metrics.TestRetrieval
else:

    def test_ai1_metric_modules_need_scipy():
        pytest.skip(
            "scipy 미설치: AI 1 추적 지표 테스트는 ai/training/ai1/requirements.txt 환경에서 실행"
        )

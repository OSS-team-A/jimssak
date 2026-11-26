#!/usr/bin/env python
"""전체 자동 학습·평가 파이프라인 진입점. 보통은 scripts/run_all.sh 로 실행한다.

python -u run_pipeline.py --profile smoke
python -u run_pipeline.py --profile standard --resume latest
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from jimssak_ai1.common import set_offline_friendly_ultralytics_env  # noqa: E402

if __name__ == "__main__":
    set_offline_friendly_ultralytics_env()
    from jimssak_ai1.orchestrator import main

    main()

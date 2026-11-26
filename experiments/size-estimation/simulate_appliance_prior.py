"""Estimate the gain of pulling washers and refrigerators toward catalog sizes.

Each catalog model in turn plays the real appliance: its size is perturbed with calibrated
errors observed on development videos (leave-one-room-out calibration), then pulled toward
the other models of the catalog, so the model being scored never helps itself.

    python experiments/size-estimation/simulate_appliance_prior.py \
        outputs/dev-eval-scene/truth.json outputs/dev-eval-scene/dav2.json \
        --metadata outputs/arkit-depth-fix/metadata.csv
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulate_bed_snap import calibrated_errors  # noqa: E402

from ai.measurement import calibration, standard_sizes  # noqa: E402
from ai.measurement.moving_contract import MeasurementResponse  # noqa: E402

AXES = ("width", "depth", "height")


def summary(errors: list[float]) -> dict:
    values = [abs(e) for e in errors]
    return {"median_relative_error": round(median(values), 3),
            "within_10_percent": round(sum(v <= 0.1 for v in values) / len(values), 3)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("truth", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261126)
    args = parser.parse_args(argv)

    errors = calibrated_errors(json.loads(args.truth.read_text()),
                               MeasurementResponse.model_validate_json(args.result.read_text()),
                               calibration.rooms_from_metadata(args.metadata))
    rng = random.Random(args.seed)
    report = {}
    for label, models in standard_sizes.APPLIANCES_MM.items():
        report[label] = {"models": len(models)}
        for kind in ("rgb", "arkit"):
            rel = {m: {a: [] for a in AXES + ("volume",)} for m in ("calibrated", "pulled")}
            for model in models:
                truth = next(size for name, size in zip(models, standard_sizes.catalog_cm(label))
                             if name == model)
                for _ in range(args.draws):
                    measured = {a: truth[a] * math.exp(rng.choice(errors[(kind, a)]))
                                for a in AXES}
                    pulled = standard_sizes.pull_to_catalog(measured, label, kind, exclude=model)
                    for name, size in (("calibrated", measured), ("pulled", pulled)):
                        for a in AXES:
                            rel[name][a].append(size[a] / truth[a] - 1)
                        volume = math.prod(size.values()) / math.prod(truth.values())
                        rel[name]["volume"].append(volume - 1)
            report[label][kind] = {m: {a: summary(v) for a, v in axes.items()}
                                   for m, axes in rel.items()}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

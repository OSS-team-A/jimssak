"""Collect the camera data and 3D annotations from selected ARKitScenes scans."""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw"
SCANS = {
    "Training": [
        "40777060",
        "40777065",
        "40777073",
        "40809740",
        "40958733",
        "40958748",
        "40958754",
        "40958764",
    ],
    "Validation": ["41069021", "41069042", "41069048"],
}


def fetch(item):
    split, scan, filename = item
    url = f"{BASE}/{split}/{scan}/{filename}"
    path = ROOT / "data" / split / scan / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_suffix(path.suffix + ".partial")
        result = subprocess.run(
            [
                "curl",
                "-fsSL",
                "--retry",
                "1",
                "--connect-timeout",
                "10",
                "--max-time",
                "40",
                "--max-filesize",
                "80000000",
                "-o",
                str(tmp),
                url,
            ],
            check=False,
        )
        if result.returncode:
            tmp.unlink(missing_ok=True)
            return {"url": url, "status": f"curl {result.returncode}"}
        tmp.replace(path)
    return {"url": url, "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size}


def main():
    items = [
        (split, scan, filename)
        for split, scans in SCANS.items()
        for scan in scans
        for filename in (
            f"{scan}_3dod_annotation.json",
            "lowres_wide.traj",
            "lowres_wide_intrinsics.zip",
        )
    ]
    with ThreadPoolExecutor(max_workers=8) as pool:
        files = list(pool.map(fetch, items))
    (ROOT / "downloads.json").write_text(json.dumps(files, indent=2) + "\n")
    print(f"Collected {sum('path' in row for row in files)} of {len(files)} files")


if __name__ == "__main__":
    main()

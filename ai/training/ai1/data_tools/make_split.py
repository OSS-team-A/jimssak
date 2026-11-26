#!/usr/bin/env python
"""방·촬영 세션·실물 그룹 단위 train/val/test 분할 (프레임 무작위 분할 금지).

입력 sessions.csv 열:
  session_id     촬영 세션 ID (영상 1편 = 1세션 권장, ASCII)
  video_path     영상 경로 (/workspace/... 기준)
  room_id        방 ID. 같은 방의 다른 촬영은 같은 room_id
  object_group   (선택) 같은 실물을 공유하는 방들의 묶음 ID. 예: 같은 집, 이사 전후 같은 가구
  role           dev  = 개발·오류 분석 전용 (언니방영상 등). test에 절대 들어가지 않음
                 auto = 자동 배정 대상
  split          (선택) 사람이 직접 정한 train/val/test. 비우면 자동 배정
  note

규칙
  - room_id 와 object_group 을 연결한 그룹 단위로만 분할한다. 같은 방·같은 실물이 분할을 넘나들면
    실패.
  - role=dev 가 포함된 그룹은 test 금지.
  - test 는 한 번 정하면 고정한다(--locked-test 로 이전 splits.csv의 test 그룹을 그대로 유지 확인).

예) python data_tools/make_split.py --sessions /workspace/jimssak_data/sessions.csv \
        --out /workspace/jimssak_data/splits.csv --val-ratio 0.2 --test-ratio 0.2 --seed 0
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jimssak_ai1.common import fail, log, read_csv, write_csv  # noqa: E402

FIELDS = [
    "session_id",
    "video_path",
    "room_id",
    "object_group",
    "role",
    "split",
    "group_id",
    "split_source",
    "note",
]


def groups_of(rows):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for r in rows:
        s = "S:" + r["session_id"]
        union(s, "R:" + r["room_id"])
        if r.get("object_group"):
            union(s, "G:" + r["object_group"])
    comp = defaultdict(list)
    for r in rows:
        comp[find("S:" + r["session_id"])].append(r)
    return list(comp.values())


def main(argv=None):
    ap = argparse.ArgumentParser(description="방·세션·실물 그룹 단위 데이터 분할")
    ap.add_argument("--sessions", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--locked-test",
        type=Path,
        default=None,
        help="이전 splits.csv. 그 test 세션 구성이 바뀌면 실패",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    rows = read_csv(args.sessions)
    if not rows:
        fail("sessions.csv가 비어 있습니다")
    seen = set()
    for r in rows:
        for k in ("session_id", "video_path", "room_id", "role"):
            if not r.get(k):
                fail(f"sessions.csv '{k}' 비어 있음: {r}")
        if r["session_id"] in seen:
            fail(f"session_id 중복: {r['session_id']}")
        seen.add(r["session_id"])
        if r["role"] not in ("dev", "auto"):
            fail(f"role은 dev/auto: {r}")
        if r.get("split") and r["split"] not in ("train", "val", "test"):
            fail(f"split은 train/val/test 또는 빈칸: {r}")

    groups = groups_of(rows)
    rng = random.Random(args.seed)
    auto_groups = []
    for gi, g in enumerate(groups):
        gid = f"grp{gi:03d}"
        manual = {r["split"] for r in g if r.get("split")}
        if len(manual) > 1:
            fail(
                f"같은 방/실물 그룹이 서로 다른 split에 "
                f"지정됨: {[(r['session_id'], r['split']) for r in g]}"
            )
        is_dev = any(r["role"] == "dev" for r in g)
        if manual:
            sp = manual.pop()
            if sp == "test" and is_dev:
                fail(
                    f"role=dev(개발용) 세션이 포함된 그룹은 test가 될 수 "
                    f"없음: {[r['session_id'] for r in g]}"
                )
            for r in g:
                r.update(split=sp, group_id=gid, split_source="manual")
        else:
            auto_groups.append((gid, g, is_dev))
    # 자동 배정: 세션 수 기준으로 test → val → train 순 채우기 (dev 그룹은 test 제외)
    rng.shuffle(auto_groups)
    n_total = len(rows)
    tgt_test, tgt_val = round(n_total * args.test_ratio), round(n_total * args.val_ratio)
    cnt = defaultdict(int)
    for r in rows:
        if r.get("split"):
            cnt[r["split"]] += 1
    for gid, g, is_dev in auto_groups:
        if not is_dev and cnt["test"] < tgt_test:
            sp = "test"
        elif cnt["val"] < tgt_val:
            sp = "val"
        else:
            sp = "train"
        cnt[sp] += len(g)
        for r in g:
            r.update(split=sp, group_id=gid, split_source=f"auto(seed={args.seed})")

    if args.locked_test:
        prev = {r["session_id"] for r in read_csv(args.locked_test) if r["split"] == "test"}
        now = {r["session_id"] for r in rows if r["split"] == "test"}
        if prev != now:
            fail(f"고정된 test 구성이 바뀜: 이전 {sorted(prev)} / 현재 {sorted(now)}")
    if args.out.exists() and not args.overwrite:
        fail(f"이미 있습니다: {args.out} (--overwrite)")
    write_csv(args.out, rows, FIELDS)
    rooms = defaultdict(set)
    for r in rows:
        rooms[r["split"]].add(r["room_id"])
    log(f"분할 저장: {args.out}")
    for sp in ("train", "val", "test"):
        log(f"  {sp}: 세션 {sum(1 for r in rows if r['split'] == sp)} | 방 {len(rooms[sp])}")
    if not rooms["test"]:
        log(
            "  주의: test 방이 없습니다 → 성능 개선을 입증할 독립 평가가 불가능(예비 결과로만 표시)"
        )


if __name__ == "__main__":
    main()

"""채점 규칙 단위 테스트. 모든 데이터는 규칙 확인용 '합성' 예시이며 실제 정답이 아니다."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jimssak_ai1.listeval import attach_review, evaluate_video, prf  # noqa: E402

V = "synthetic_video"


def gt(item_id, cat, box_fit="no_fit"):
    return {"video_id": V, "item_id": item_id, "category": cat, "box_fit": box_fit}


def pred(pid, cat, t, status="confirmed", tracks=None, nf=None, box_fit="unknown"):
    return {
        "video_id": V,
        "pred_item_id": pid,
        "category": cat,
        "first_time_sec": t,
        "status": status,
        "track_ids": tracks or "",
        "track_n_frames": nf or "",
        "box_fit": box_fit,
    }


def rv(pid, g, extra=""):
    return {"video_id": V, "pred_item_id": pid, "gt_item_id": g, "extra_gt_item_ids": extra}


class TestListEval(unittest.TestCase):
    def test_duplicate_and_miss_same_total(self):
        """의자 A를 두 번 세고 B를 놓치면 총수는 같아도 TP1 FP1(중복) FN1."""
        g = [gt("chairA", "chair"), gt("chairB", "chair")]
        p = [pred("T00001", "chair", 1.0), pred("T00002", "chair", 5.0)]
        r = [rv("T00001", "chairA"), rv("T00002", "chairA")]
        pr, _ = attach_review(p, r)
        res = evaluate_video(V, g, pr, r)
        self.assertEqual((res.tp, res.fp_duplicate, res.fn), (1, 1, 1))
        self.assertEqual(res.per_cat["chair"]["pred"], 2)  # 수량 오차는 0 이지만 F1은 낮음
        p_, r_, f_ = prf(res.tp, res.fp, res.fn)
        self.assertAlmostEqual(f_, 0.5)

    def test_misclass_and_nonexistent(self):
        g = [gt("desk1", "desk")]
        p = [pred("T00001", "table", 1.0), pred("T00002", "sofa", 2.0)]
        r = [rv("T00001", "desk1"), rv("T00002", "")]
        pr, _ = attach_review(p, r)
        res = evaluate_video(V, g, pr, r, unsupported_categories={"desk"})
        self.assertEqual((res.tp, res.fp_misclass, res.fp_nonexistent, res.fn), (0, 1, 1, 1))
        fn_case = [c for c in res.cases if c["outcome"].startswith("FN")][0]
        self.assertIn("미지원 품목", fn_case["auto_hint"])
        self.assertIn("클래스 혼동", fn_case["auto_hint"])

    def test_needs_review_not_tp(self):
        g = [gt("bed1", "bed")]
        p = [pred("T00001", "bed", 1.0, status="needs_review")]
        r = [rv("T00001", "bed1")]
        pr, _ = attach_review(p, r)
        res = evaluate_video(V, g, pr, r)
        self.assertEqual((res.tp, res.fp, res.fn, res.needs_review_target), (0, 0, 1, 1))

    def test_out_of_scope_and_unknown(self):
        g = [gt("chair1", "chair", "fits"), gt("sofa1", "sofa", "unknown"), gt("bed1", "bed")]
        p = [pred("T1", "chair", 1.0), pred("T2", "sofa", 2.0), pred("T3", "bed", 3.0)]
        r = [rv("T1", "chair1"), rv("T2", "sofa1"), rv("T3", "bed1")]
        pr, _ = attach_review(p, r)
        res = evaluate_video(V, g, pr, r, mode="list")
        self.assertEqual(
            (res.tp, res.fp, res.fn, res.out_of_scope, res.n_gt_unknown), (1, 0, 0, 2, 1)
        )

    def test_box_mode(self):
        g = [gt("chair1", "chair", "fits"), gt("bed1", "bed")]
        p = [pred("T1", "chair", 1.0, box_fit="no_fit"), pred("T3", "bed", 3.0, box_fit="fits")]
        r = [rv("T1", "chair1"), rv("T3", "bed1")]
        pr, _ = attach_review(p, r)
        res = evaluate_video(V, g, pr, r, mode="box")
        self.assertEqual((res.tp, res.fp_box_fit, res.fn), (0, 1, 1))

    def test_relinked_group_derived_from_tracks(self):
        """B2 묶음: 같은 실물 트랙 둘을 합치면 중복 FP가 사라진다.

        다른 실물을 합치면 잘못된 병합으로 센다.
        """
        g = [gt("sofaA", "sofa"), gt("sofaB", "sofa")]
        track_review = [
            rv("T00001", "sofaA"),
            rv("T00002", "sofaA"),
            rv("T00003", "sofaB"),
            rv("T00004", "sofaB"),
        ]
        # 올바른 병합: T1+T2 → G1, T3 단독, T4 단독(중복으로 남음)
        p = [
            pred("G00001", "sofa", 1.0, tracks="1;2", nf="30;10"),
            pred("T00003", "sofa", 5.0, tracks="3"),
            pred("T00004", "sofa", 9.0, tracks="4"),
        ]
        pr, w = attach_review(p, track_review)
        self.assertEqual(w, [])
        res = evaluate_video(V, g, pr, track_review)
        self.assertEqual((res.tp, res.fp_duplicate, res.fn, res.wrong_merge), (2, 1, 0, 0))
        # 잘못된 병합: T2+T3 → 서로 다른 실물
        p2 = [
            pred("T00001", "sofa", 1.0, tracks="1"),
            pred("G00001", "sofa", 3.0, tracks="2;3", nf="5;40"),
            pred("T00004", "sofa", 9.0, tracks="4"),
        ]
        pr2, _ = attach_review(p2, track_review)
        g1 = [x for x in pr2 if x["pred_item_id"] == "G00001"][0]
        self.assertEqual(g1["_gt"], "sofaB")  # 프레임이 많은 트랙(T3)의 실물이 대표
        self.assertEqual(g1["_extra"], ["sofaA"])
        res2 = evaluate_video(V, g, pr2, track_review)
        self.assertEqual(res2.wrong_merge, 1)

    def test_unreviewed_pred(self):
        g = [gt("bed1", "bed")]
        p = [pred("T00009", "bed", 1.0)]
        pr, w = attach_review(p, [])
        self.assertTrue(w)
        res = evaluate_video(V, g, pr, [])
        self.assertEqual((res.fp_unreviewed, res.f1_judgeable), (1, False))


if __name__ == "__main__":
    unittest.main()

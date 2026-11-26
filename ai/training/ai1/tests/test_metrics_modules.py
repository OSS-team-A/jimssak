"""추적 지표·전후 매칭·AUROC 단위 테스트 (합성 입력, 정답이 분명한 경우만)."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jimssak_ai1.matching.before_after import match_items, score_scenario  # noqa: E402
from jimssak_ai1.reidentification.train_reid import auroc, retrieval_metrics  # noqa: E402
from jimssak_ai1.tracking.mot_metrics import compute_mot, item_level  # noqa: E402

B1 = np.array([0, 0, 10, 10.0])
B2 = np.array([50, 50, 60, 60.0])


class TestMot(unittest.TestCase):
    def test_perfect(self):
        fr = [{"gt": [(1, B1), (2, B2)], "tr": [(7, B1), (8, B2)]} for _ in range(10)]
        m = compute_mot(fr)
        self.assertAlmostEqual(m["HOTA"], 1.0, places=4)
        self.assertAlmostEqual(m["IDF1"], 1.0, places=4)
        self.assertEqual(m["IDSW"], 0)
        self.assertAlmostEqual(m["MOTA"], 1.0, places=4)

    def test_id_switch(self):
        fr = [{"gt": [(1, B1)], "tr": [(7, B1)]} for _ in range(5)] + [
            {"gt": [(1, B1)], "tr": [(9, B1)]} for _ in range(5)
        ]
        m = compute_mot(fr)
        self.assertEqual(m["IDSW"], 1)
        self.assertAlmostEqual(m["IDF1"], 0.5, places=4)
        self.assertLess(m["AssA"], 1.0)
        self.assertAlmostEqual(m["DetA"], 1.0, places=4)

    def test_item_level_duplicate_and_reappearance(self):
        # 정답 1이 0~4, 8~9 프레임에 보임(재등장). 추적은 두 트랙(7, 9)으로 나뉨
        fr = []
        for t in range(10):
            g = [(1, B1)] if t < 5 or t >= 8 else []
            tr = [(7, B1)] if t < 5 else ([(9, B1)] if t >= 8 else [])
            fr.append({"gt": g, "tr": tr})
        res = item_level(fr, {7: 7, 9: 9}, {1: "chair"}, {7: "chair", 9: "chair"}, {1}, {7, 9})
        self.assertEqual((res["TP"], res["duplicates"]), (1, 1))
        self.assertEqual(res["reappearance_link_accuracy"], 0.0)
        res2 = item_level(
            fr, {7: 7, 9: 7}, {1: "chair"}, {7: "chair"}, {1}, {7}
        )  # 재등장 연결 성공
        self.assertEqual(
            (res2["TP"], res2["duplicates"], res2["reappearance_link_accuracy"]), (1, 0, 1.0)
        )


class TestMatching(unittest.TestCase):
    def test_missing_and_unexpected(self):
        def e(*v):
            return np.array(v, float) / np.linalg.norm(v)

        before = [
            {"item_id": "b1", "class": "chair", "emb": e(1, 0)},
            {"item_id": "b2", "class": "chair", "emb": e(0, 1)},
        ]
        after = [
            {"item_id": "a1", "class": "chair", "emb": e(1, 0.05)},
            {"item_id": "a3", "class": "sofa", "emb": e(0, 1)},
        ]
        res = match_items(before, after, thr=0.9)
        self.assertEqual(res["before"]["b1"]["status"], "matched")
        self.assertEqual(res["before"]["b2"]["status"], "missing_candidate")
        self.assertTrue(res["before"]["b2"]["review_required"])
        self.assertEqual(res["after"]["a3"]["status"], "unexpected_item")
        sc = score_scenario(res, {"b2"}, {"a3"}, {"b1": "a1"})
        self.assertEqual((sc["missing_tp"], sc["false_missing"], sc["correct_pairs"]), (1, 0, 1))


class TestRetrieval(unittest.TestCase):
    def test_auroc_and_recall(self):
        self.assertAlmostEqual(auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])), 1.0)
        self.assertAlmostEqual(auroc(np.array([0.5, 0.5]), np.array([0, 1])), 0.5)
        z = np.array([[1, 0], [0.99, 0.1], [0, 1], [0.1, 0.99]], float)
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        m = retrieval_metrics(z, ["a", "a", "b", "b"])
        self.assertEqual(m["recall@1"], 1.0)


if __name__ == "__main__":
    unittest.main()

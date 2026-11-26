"""품목 분류 체계와 모델 클래스 → 품목 코드 매핑."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .common import PROJECT_ROOT, fail, load_yaml

DEFAULT_TAXONOMY = PROJECT_ROOT / "configs" / "taxonomy.yaml"


@dataclass
class Taxonomy:
    categories: list[dict]

    @property
    def codes(self) -> list[str]:
        return [c["code"] for c in self.categories]

    def ko(self, code: str) -> str:
        for c in self.categories:
            if c["code"] == code:
                return c.get("ko") or code
        return code

    def model_class_to_code(self, model_names: dict[int, str]) -> tuple[dict[int, str], str]:
        """모델 클래스 id → 품목 코드.

        - 모델 클래스명이 taxonomy code와 같으면(파인튜닝 모델) 그대로 매핑: mode='code'
        - 아니면 taxonomy의 coco 열과 정확히 같은 이름만 매핑: mode='coco'
        비슷한 클래스로 추정 매핑하지 않는다. 매핑되지 않은 클래스는 결과에서 'unmapped'로 남는다.
        """
        names = {int(k): str(v) for k, v in model_names.items()}
        codes = set(self.codes)
        if names and all(v in codes for v in names.values()):
            return {k: v for k, v in names.items()}, "code"
        coco_to_code = {c["coco"]: c["code"] for c in self.categories if c.get("coco")}
        mapping = {k: coco_to_code[v] for k, v in names.items() if v in coco_to_code}
        return mapping, "coco"

    def support_table(self, model_names: dict[int, str]) -> list[dict]:
        """품목별 지원 여부 표.

        미지원 품목을 다른 클래스로 몰래 대체하지 않았음을 보여주기 위함.
        """
        mapping, mode = self.model_class_to_code(model_names)
        code_to_cls = {}
        for cid, code in mapping.items():
            code_to_cls.setdefault(code, []).append(model_names[cid])
        rows = []
        for c in self.categories:
            if c["code"] == "other":
                continue
            cls = code_to_cls.get(c["code"], [])
            rows.append(
                {
                    "category": c["code"],
                    "ko": c.get("ko", ""),
                    "supported": "yes" if cls else "no(미지원)",
                    "model_class": ";".join(cls),
                    "mapping_mode": mode,
                    "note": c.get("note", ""),
                }
            )
        return rows


def load_taxonomy(path: Path | None = None) -> Taxonomy:
    data = load_yaml(path or DEFAULT_TAXONOMY)
    cats = data.get("categories")
    if not isinstance(cats, list) or not cats:
        fail(f"taxonomy 파일에 categories 목록이 없습니다: {path}")
    seen = set()
    for c in cats:
        if "code" not in c:
            fail(f"taxonomy 항목에 code가 없습니다: {c}")
        if c["code"] in seen:
            fail(f"taxonomy code 중복: {c['code']}")
        seen.add(c["code"])
    return Taxonomy(categories=cats)

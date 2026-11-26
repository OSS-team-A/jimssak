from contextlib import contextmanager

from psycopg.types.json import Jsonb

from ..db import Database
from ..schemas.inventory import Inventory
from .analysis import AnalysisJob


def load_inventory(db: Database, job: AnalysisJob | None) -> Inventory:
    if job is None or job.status != "completed":
        return Inventory()
    with db.connect() as conn:
        row = conn.execute(
            "SELECT items, confirmed FROM move_inventories WHERE analysis_id = %s",
            (job.analysis_id,),
        ).fetchone()
    if row is None:
        return Inventory()
    items = row["items"]
    return Inventory(
        status="confirmed" if row["confirmed"] else "draft",
        items=items, item_count=len({i["label_ko"] for i in items}),
        total_quantity=sum(i["quantity"] for i in items),
    )


@contextmanager
def edit_inventory(db: Database, analysis_id: str):
    # Serialize edits so two requests changing different items keep both changes.
    with db.connect() as conn:
        row = conn.execute(
            "SELECT items, confirmed FROM move_inventories WHERE analysis_id = %s FOR UPDATE",
            (analysis_id,),
        ).fetchone()
        yield row
        conn.execute(
            "UPDATE move_inventories SET items = %s, confirmed = %s WHERE analysis_id = %s",
            (Jsonb(row["items"]), row["confirmed"], analysis_id),
        )

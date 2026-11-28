from contextlib import contextmanager, nullcontext

from psycopg import Connection
from psycopg.types.json import Jsonb

from ..db import Database
from ..schemas.inventory import Inventory
from .analysis import AnalysisJob


def load_inventory(
    db: Database, job: AnalysisJob | None, *, conn: Connection | None = None,
    for_share: bool = False,
) -> Inventory:
    if job is None or job.status != "completed":
        return Inventory()
    query = "SELECT items, confirmed FROM move_inventories WHERE analysis_id = %s"
    if for_share:
        query += " FOR SHARE"
    with db.connect() if conn is None else nullcontext(conn) as active:
        row = active.execute(query, (job.analysis_id,)).fetchone()
    return inventory_from_row(row)


def inventory_from_row(row: dict | None) -> Inventory:
    if row is None:
        return Inventory()
    items = row["items"]
    return Inventory(
        status="confirmed" if row["confirmed"] else "draft",
        items=items, item_count=len({i["label_ko"] for i in items}),
        total_quantity=sum(i["quantity"] for i in items),
    )


@contextmanager
def edit_inventory(db: Database, analysis_id: str, *, conn: Connection | None = None):
    # Serialize edits so two requests changing different items keep both changes.
    with db.connect() if conn is None else nullcontext(conn) as active:
        row = active.execute(
            "SELECT items, confirmed FROM move_inventories WHERE analysis_id = %s FOR UPDATE",
            (analysis_id,),
        ).fetchone()
        yield row
        if row is None:
            return
        active.execute(
            "UPDATE move_inventories SET items = %s, confirmed = %s WHERE analysis_id = %s",
            (Jsonb(row["items"]), row["confirmed"], analysis_id),
        )

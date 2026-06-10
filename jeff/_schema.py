"""Shared schema-drift guard for the per-peer Postgres stores.

Every store builds its table with ``CREATE TABLE IF NOT EXISTS`` and offers a
destructive ``reset()`` — there is no migration framework. That leaves one sharp
edge: if a table from an *older* build already exists, ``CREATE TABLE IF NOT
EXISTS`` is a no-op, so a column the new code added never appears. The drift is
silent until the first query touches the missing column, which then fails at
runtime with ``UndefinedColumn`` deep inside a turn (this bit live — ticket
dbafb5ea: ``proactive_state`` lacked ``last_send_at`` after a redeploy onto an
old table from the reverted inner-life branch).

Two-part discipline closes the hole:

1. **Additive migrations** — when a store gains a column, emit an idempotent
   ``ALTER TABLE <t> ADD COLUMN IF NOT EXISTS <col> <type>`` next to the CREATE
   (see ``proactive.py``). It self-applies on already-deployed tables and is a
   no-op on fresh ones; both shapes converge. Postgres has no
   ``ADD CONSTRAINT IF NOT EXISTS``, so CHECK/UNIQUE constraints stay enforced
   in-app rather than retro-fitted here.
2. **A startup guard (this module)** — after running its DDL, each store calls
   :func:`assert_columns` with the column set its queries rely on. On a drifted
   DB the guard raises :class:`SchemaDriftError` at ``create()`` time — i.e. at
   pod startup / deploy, naming the table and the missing column(s) — instead of
   letting the gap surface mid-turn. Mirrors the embed-dim guard's fail-loud
   stance (``Memory._guard_embed_dim``): drift is caught at deploy, not first use.

The guard only *detects*; it never alters. Fixing additive drift is the one-line
ALTER above; a shape change that can't be expressed additively is a ``reset()``
/ ``reset-memory`` rebuild.
"""

from __future__ import annotations

from collections.abc import Iterable


class SchemaDriftError(RuntimeError):
    """A live table is missing column(s) the code expects (see module docstring)."""

    def __init__(self, table: str, missing: Iterable[str]) -> None:
        self.table = table
        self.missing = sorted(set(missing))
        cols = ", ".join(self.missing)
        super().__init__(
            f"schema drift: table {table!r} is missing column(s) [{cols}] that the "
            f"code expects. An older {table} predates this column set and "
            "CREATE TABLE IF NOT EXISTS left it untouched. Fix: add an idempotent "
            f"`ALTER TABLE {table} ADD COLUMN IF NOT EXISTS ...` for the new column "
            "(additive change), or `python -m jeff reset-memory --yes` to rebuild."
        )


async def assert_columns(cur, table: str, expected: Iterable[str]) -> None:
    """Fail loudly at startup if ``table`` lacks any of the ``expected`` columns.

    Call it at the end of a store's ``_init_schema`` — after the CREATE and any
    ADD COLUMN migrations — so on a fresh or properly-migrated DB it's a silent
    no-op, and on a drifted one it raises :class:`SchemaDriftError` naming the
    gap. If the table doesn't exist yet (``to_regclass`` is NULL) there's nothing
    to guard — the CREATE that runs before us owns that case.
    """
    want = frozenset(expected)
    await cur.execute("SELECT to_regclass(%s)", (table,))
    row = await cur.fetchone()
    if row is None or row[0] is None:
        return
    await cur.execute(
        "SELECT attname FROM pg_attribute "
        "WHERE attrelid = to_regclass(%s) AND attnum > 0 AND NOT attisdropped",
        (table,),
    )
    have = {r[0] for r in await cur.fetchall()}
    missing = want - have
    if missing:
        raise SchemaDriftError(table, missing)

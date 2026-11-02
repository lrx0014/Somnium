"""Alembic-managed database schema migrations for Hippocore (M2.1).

Uses `alembic.runtime.migration.MigrationContext` to manage and query database
revisions in the standard `alembic_version` table, with explicit forward
(`upgrade_database`) and reverse (`downgrade_database`) migration operations.
"""

from __future__ import annotations

from alembic.runtime.migration import MigrationContext
from sqlalchemy import Column, Engine, MetaData, String, Table, inspect, text

from somnium.hippocore.schema import (
    deletion_requests,
    memory_access,
    memory_events,
    memory_facts,
    memory_revisions,
    memory_sources,
    metadata,
    routing_decisions,
    tenant_state,
)

INITIAL_REVISION = "0001_initial_hippocore"
HEAD_REVISION = INITIAL_REVISION

_ALEMBIC_META = MetaData()
alembic_version = Table(
    "alembic_version",
    _ALEMBIC_META,
    Column("version_num", String(64), primary_key=True, nullable=False),
)

_ALL_CORE_TABLES = (
    memory_events,
    memory_facts,
    memory_revisions,
    memory_sources,
    memory_access,
    deletion_requests,
    tenant_state,
    routing_decisions,
)


def current_revision(engine: Engine) -> str | None:
    """Return the current Alembic migration revision of `engine`, or None if uninitialized."""
    with engine.connect() as conn:
        ctx = MigrationContext.configure(conn)
        return ctx.get_current_revision()


def upgrade_database(engine: Engine, target: str = "head") -> str:
    """Apply forward migrations up to `target` ('head' or '0001_initial_hippocore')."""
    if target not in ("head", INITIAL_REVISION):
        raise ValueError(f"Unknown migration target '{target}'.")

    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        current = ctx.get_current_revision()
        if current == INITIAL_REVISION:
            return INITIAL_REVISION

        inspector = inspect(conn)
        existing_tables = set(inspector.get_table_names())

        if "alembic_version" not in existing_tables:
            alembic_version.create(conn, checkfirst=True)

        for table in _ALL_CORE_TABLES:
            if table.name not in existing_tables:
                table.create(conn, checkfirst=True)

        # Create SQLite FTS5 virtual table when running on SQLite
        if conn.dialect.name == "sqlite" and "memory_fts" not in existing_tables:
            conn.execute(
                text(
                    "CREATE VIRTUAL TABLE memory_fts USING fts5("
                    "memory_id UNINDEXED, "
                    "tenant_id UNINDEXED, "
                    "scope_kind UNINDEXED, "
                    "scope_id UNINDEXED, "
                    "content, "
                    "tags"
                    ")"
                )
            )

        conn.execute(alembic_version.delete())
        conn.execute(alembic_version.insert().values(version_num=INITIAL_REVISION))

    return INITIAL_REVISION


def downgrade_database(engine: Engine, target: str = "base") -> None:
    """Revert migrations down to `target` ('base')."""
    if target != "base":
        raise ValueError(f"Unknown downgrade target '{target}'. Expected 'base'.")

    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        current = ctx.get_current_revision()
        if current is None:
            return

        inspector = inspect(conn)
        existing_tables = set(inspector.get_table_names())

        if conn.dialect.name == "sqlite" and "memory_fts" in existing_tables:
            conn.execute(text("DROP TABLE IF EXISTS memory_fts"))

        for table in reversed(_ALL_CORE_TABLES):
            if table.name in existing_tables:
                table.drop(conn, checkfirst=True)

        if "alembic_version" in existing_tables:
            conn.execute(alembic_version.delete())


__all__ = [
    "HEAD_REVISION",
    "INITIAL_REVISION",
    "current_revision",
    "downgrade_database",
    "metadata",
    "upgrade_database",
]

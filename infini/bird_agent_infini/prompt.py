"""Prompt builder for BIRD-Critic Flash (PostgreSQL) debugging tasks."""

from __future__ import annotations

# Statement separator the agent must place between multiple SQL statements in
# the deliverable file. An explicit marker avoids naive semicolon splitting,
# which would break dollar-quoted ($$ ... $$) PL/pgSQL bodies.
SQL_SPLIT_MARKER = "-- [BIRD_SPLIT]"


def _as_text(value) -> str:
    """Join a str-or-list jsonl field into displayable text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n\n".join(str(v).strip() for v in value if str(v).strip())
    return str(value)


def deliverable_name(instance_id) -> str:
    """Workspace file the agent must produce for this instance."""
    return f"{instance_id}.sql"


def build_prompt(instance_id, query, issue_sql) -> str:
    """Build the newTask text for one BIRD-Critic Flash instance.

    Args:
        instance_id: instance identifier from the jsonl (int for Flash).
        query: natural-language user issue description.
        issue_sql: the problematic SQL (str or list of str).
    """
    issue_sql_text = _as_text(issue_sql)
    sql_file = deliverable_name(instance_id)

    return f"""You are a PostgreSQL SQL-debugging agent. A user ran into an issue with their SQL against the PostgreSQL database that is already attached to this task. Diagnose the issue and produce the corrected SQL.

<objective>
Produce exactly one deliverable file in the task workspace:
1. `{sql_file}` — the corrected SQL that resolves the user's issue.
</objective>

<rules>
- You MUST use Infinity SQL (via `execute_infinity_sql`) to explore the attached database (schema, sample rows) and to verify that your corrected SQL actually resolves the user's issue. Do NOT fabricate results.
- The corrected SQL in `{sql_file}` must be plain PostgreSQL SQL — it will be executed verbatim against a PostgreSQL 14 database by an automated grader. Do NOT include markdown fences, comments explaining the fix, or any prose.
- If the fix requires MULTIPLE SQL statements executed in order, separate consecutive statements with a line containing exactly `{SQL_SPLIT_MARKER}` (each statement may itself span multiple lines). If a single statement suffices, write just that statement without the marker.
- Do NOT add a trailing semicolon-separated dump of alternatives; the file must contain only the final answer statements.
- Preserve the user's intent exactly: fix what is broken, do not change requested columns, filters, ordering, or add extra output the user did not ask for.
- If the issue is ambiguous, pick the most reasonable interpretation based on what you observe in the actual database.
- Never stop early: keep iterating until `{sql_file}` exists and its content is verified against the database.
</rules>

<user_issue>
{query}
</user_issue>

<problematic_sql>
{issue_sql_text}
</problematic_sql>
"""

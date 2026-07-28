"""Harvest corrected SQL from a downloaded task workspace into pred_sqls jsonl."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path

from bird_agent_infini.prompt import SQL_SPLIT_MARKER, deliverable_name

logger = logging.getLogger("bird_agent_infini")

# harvest results are appended from concurrent workers; serialize writes
_pred_file_lock = threading.Lock()

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*$")


def find_first(root: Path, name: str) -> Path | None:
    """Recursively find the first file called *name* under *root*."""
    for cur, _dirs, files in os.walk(root):
        if name in files:
            return Path(cur) / name
    return None


def extract_pred_sqls(sql_text: str) -> list[str]:
    """Turn the deliverable file content into a ``pred_sqls`` list.

    Statements are split on the explicit ``-- [BIRD_SPLIT]`` marker requested
    in the prompt; without a marker the whole file is one statement. Naive
    semicolon splitting is deliberately avoided (dollar-quoted PL/pgSQL bodies
    contain semicolons). Stray markdown fences are stripped defensively in
    case the agent disobeyed the plain-SQL instruction.
    """
    lines = [ln for ln in sql_text.splitlines() if not _FENCE_RE.match(ln.strip())]
    cleaned = "\n".join(lines)

    parts = re.split(re.escape(SQL_SPLIT_MARKER), cleaned)
    sqls = [p.strip() for p in parts if p.strip()]
    return sqls


def harvest_workspace(instance_id, workspace_dir: Path) -> list[str] | None:
    """Locate the deliverable in *workspace_dir* and return its pred_sqls.

    Returns None when the deliverable file is missing or empty.
    """
    name = deliverable_name(instance_id)
    src = find_first(workspace_dir, name)
    if src is None:
        logger.warning("[miss ] %s: deliverable %s not found in workspace",
                       instance_id, name)
        return None
    text = src.read_text(encoding="utf-8", errors="replace")
    sqls = extract_pred_sqls(text)
    if not sqls:
        logger.warning("[miss ] %s: deliverable %s is empty", instance_id, name)
        return None
    return sqls


def upsert_pred(pred_path: Path, instance_id, pred_sqls: list[str]) -> None:
    """Insert-or-replace one prediction row in the pred jsonl (by instance_id).

    The file stays sorted by insertion order; replacing keeps the original
    position. Writes are serialized so concurrent workers can call this
    directly after each task finishes.
    """
    with _pred_file_lock:
        rows: list[dict] = []
        if pred_path.exists():
            with open(pred_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))

        replaced = False
        for row in rows:
            if row.get("instance_id") == instance_id:
                row["pred_sqls"] = pred_sqls
                replaced = True
                break
        if not replaced:
            rows.append({"instance_id": instance_id, "pred_sqls": pred_sqls})

        pred_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = pred_path.with_suffix(".jsonl.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp_path.replace(pred_path)


def load_pred_ids(pred_path: Path) -> set:
    """Return the set of instance_ids already present in the pred jsonl."""
    ids: set = set()
    if not pred_path.exists():
        return ids
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("pred_sqls"):
                ids.add(row.get("instance_id"))
    return ids

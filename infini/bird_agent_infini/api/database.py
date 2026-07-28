"""InfiniSynapse API helpers used by the BIRD-Critic runner.

Trimmed port of Spider2's ``spider_agent_infini.api.database`` keeping only
what the Flash runner needs: data-source lookup, task submission, task
polling and workspace download. No setup / registration helpers.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import requests

from bird_agent_infini.api.client import DEFAULT_TIMEOUT, InfiniClient, unwrap

logger = logging.getLogger("bird_agent_infini")


def normalize_database_name(name: str) -> str:
    """Normalize a string for use as an InfiniSynapse data source ``name``.

    InfiniSynapse rejects ``-`` in registered database names, and we lowercase
    to keep lookups deterministic across casing variants. Same rule as the
    Spider2 setup pipeline, so sources registered there are matched here.
    """
    return name.lower().replace("-", "_")


def list_databases(
    name: str | None = None,
    type: str | None = None,
    enabled: int | None = None,
    source: str = "all",
    page: int = 1,
    page_size: int = 10000,
    credential_path: str | os.PathLike | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[dict[str, Any]]:
    """List databases via `GET /api/ai_database/list`.

    Returns the `items` array from the paginated response. Defaults to a large
    `pageSize` so that a single call typically returns everything.
    """
    params: dict[str, Any] = {"page": page, "pageSize": page_size, "source": source}
    if name is not None:
        params["name"] = name
    if type is not None:
        params["type"] = type
    if enabled is not None:
        params["enabled"] = enabled

    client = InfiniClient(credential_path=credential_path, timeout=timeout)
    resp = client.get("/api/ai_database/list", params=params)
    data = unwrap(resp.json())
    if isinstance(data, dict) and "items" in data:
        return list(data["items"] or [])
    if isinstance(data, list):
        return data
    return []


def select_databases_by_postgres_db_id(
    db_id: str,
    credential_path: str | os.PathLike | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[dict[str, Any]]:
    """Return PostgreSQL data sources whose ``name`` matches the given ``db_id``.

    Mirrors Spider2's ``select_databases_by_sqlite_db_id`` but for PostgreSQL
    and WITHOUT any global enable/disable toggling — callers scope the task
    via ``databaseIds`` on ``newTask``, which is concurrency-safe.

    The server may register the type as ``postgres`` or ``postgresql``
    depending on version, so both are queried.
    """
    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for type_name in ("postgres", "postgresql"):
        try:
            batch = list_databases(
                type=type_name,
                credential_path=credential_path,
                timeout=timeout,
            )
        except requests.RequestException as e:
            logger.warning("list_databases(type=%r) failed: %s", type_name, e)
            continue
        for item in batch:
            if not isinstance(item, dict):
                continue
            iid = str(item.get("id") or "")
            if iid and iid in seen_ids:
                continue
            seen_ids.add(iid)
            items.append(item)

    target_name = normalize_database_name(db_id)
    matching = [
        item
        for item in items
        if normalize_database_name(str(item.get("name") or "")) == target_name
    ]
    return matching


def list_available_engines(
    credential_path: str | os.PathLike | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[dict[str, Any]]:
    """List available InfiniSQL engines via `GET /api/ai_byzer/available`.

    Each item typically includes ``id``, ``name``, and ``url``; pass an id as
    ``engineId`` on ``newTask``.
    """
    client = InfiniClient(credential_path=credential_path, timeout=timeout)
    resp = client.get("/api/ai_byzer/available")
    data = unwrap(resp.json())
    if isinstance(data, dict) and "items" in data:
        return list(data["items"] or [])
    if isinstance(data, list):
        return data
    return []


def new_task(
    text: str,
    task_id: str | None = None,
    database_ids: list[str] | None = None,
    engine_id: str | None = None,
    command_id: str | None = None,
    client_message_id: str | None = None,
    extra: dict[str, Any] | None = None,
    credential_path: str | os.PathLike | None = None,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Create a new InfiniSynapse task via `POST /api/ai/message`.

    Args:
        text: first user message / task description.
        task_id: existing taskId for idempotency; auto-generated if omitted.
        database_ids: per-task data source ids (sent as ``databaseIds``). When
            provided, the task uses exactly these sources regardless of which
            sources are globally enabled — this is what lets concurrent tasks
            target different databases without a global enable/disable toggle.
        engine_id: per-task InfiniSQL engine id (sent as ``engineId``).
        command_id / client_message_id: idempotency identifiers; auto-filled
            if omitted.
        extra: additional fields to merge into the request body.

    Returns:
        The unwrapped server response (typically the ack).
    """
    tid = task_id or str(uuid.uuid4())
    payload: dict[str, Any] = {
        "type": "newTask",
        "taskId": tid,
        "text": text,
        "commandId": command_id or str(uuid.uuid4()),
        "clientMessageId": client_message_id or str(uuid.uuid4()),
    }
    if database_ids:
        payload["databaseIds"] = list(database_ids)
    if engine_id:
        payload["engineId"] = engine_id
    if extra:
        payload.update(extra)

    client = InfiniClient(credential_path=credential_path, timeout=timeout)
    resp = client.post("/api/ai/message", json_body=payload)
    return unwrap(resp.json())


def get_task_data(
    task_id: str,
    credential_path: str | os.PathLike | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Fetch a task's full data via `GET /api/ai_task/tasks?taskId=...`.

    Returns the unwrapped payload, typically
    ``{"taskInfo": {..., "status": "running"|"completed"|...},
        "messages": [...], "isRunning": bool}``.
    """
    client = InfiniClient(credential_path=credential_path, timeout=timeout)
    resp = client.get("/api/ai_task/tasks", params={"taskId": task_id})
    return unwrap(resp.json())


def download_task_zip(
    task_id: str,
    dest: str | os.PathLike,
    credential_path: str | os.PathLike | None = None,
    timeout: float = 600.0,
    chunk_size: int = 256 * 1024,
) -> str:
    """Download the whole task workspace as a ZIP.

    Calls `GET /api/ai_task/downloadZip?taskId=<id>`. Returns the local
    absolute path of the saved zip.
    """
    client = InfiniClient(credential_path=credential_path, timeout=timeout)
    url = f"{client.api_url}/api/ai_task/downloadZip"

    dest_path = Path(dest)
    if dest_path.exists() and dest_path.is_dir():
        dest_path = dest_path / f"{task_id}.zip"
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(
        url,
        params={"taskId": task_id},
        headers=client._headers(),
        stream=True,
        timeout=timeout,
    ) as resp:
        resp.raise_for_status()
        with open(dest_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    fh.write(chunk)
    return str(dest_path.resolve())


def _last_non_partial_message(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the last finalized (non-partial) message, if any."""
    for m in reversed(messages or []):
        if not isinstance(m, dict):
            continue
        if m.get("partial"):
            continue
        return m
    return None


def _is_terminal_message(
    msg: dict[str, Any] | None,
    *,
    terminal_on_any_ask: bool = True,
) -> bool:
    """Detect a terminal "task is done" message from the message stream.

    The server keeps the Infini instance registered after
    ``attempt_completion`` (``keepRegisteredOnAskExit=true``), so ``isRunning``
    stays True and ``taskInfo.status`` stays "running" until the runtime is
    parked. The authoritative completion signal is the message stream itself.
    """
    if not msg:
        return False
    mtype = msg.get("type")
    if mtype == "say" and msg.get("say") == "completion_result":
        return True
    if mtype == "ask":
        ask = msg.get("ask")
        if ask == "completion_result":
            return True
        if not terminal_on_any_ask:
            return False
        # Any finalized ask other than the user-driven resume prompts means the
        # agent has handed control back.
        if ask and ask not in ("resume_task", "resume_completed_task"):
            return True
    return False


def wait_for_task(
    task_id: str,
    poll_interval: float = 3.0,
    max_wait: float = 1800.0,
    terminal_on_any_ask: bool = True,
    credential_path: str | os.PathLike | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Poll a task until it has handed control back to the caller.

    A task is considered finished when either ``taskInfo.status`` is terminal
    (``completed``/``failed``/``cancelled``/``error``) or the last finalized
    message is a ``completion_result`` / non-resume ``ask``.

    Raises ``TimeoutError`` when ``max_wait`` seconds elapse first.
    Returns the final ``get_task_data`` payload.
    """
    terminal_status = {"completed", "failed", "cancelled", "canceled", "error"}
    start = time.time()
    seen_alive = False
    poll_failures = 0

    while True:
        elapsed = time.time() - start
        if elapsed > max_wait:
            raise TimeoutError(
                f"wait_for_task({task_id}) exceeded {max_wait}s; "
                f"last poll failures={poll_failures}, seen_alive={seen_alive}"
            )

        try:
            data = get_task_data(
                task_id, credential_path=credential_path, timeout=timeout
            )
            poll_failures = 0
        except requests.RequestException as e:
            poll_failures += 1
            logger.warning(
                "wait_for_task(%s): poll failed (#%d, elapsed=%.0fs): %s; "
                "retrying in %.1fs",
                task_id, poll_failures, elapsed, e, poll_interval,
            )
            time.sleep(poll_interval)
            continue

        is_running = bool(data.get("isRunning"))
        info = data.get("taskInfo") or {}
        status = ""
        if isinstance(info, dict):
            status = str(info.get("status") or "").lower()
        messages = data.get("messages") or []

        if is_running or status or messages or info:
            seen_alive = True

        if not seen_alive:
            time.sleep(poll_interval)
            continue

        if status in terminal_status:
            return data

        if _is_terminal_message(
            _last_non_partial_message(messages),
            terminal_on_any_ask=terminal_on_any_ask,
        ):
            return data

        time.sleep(poll_interval)

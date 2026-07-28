"""Run the BIRD-Critic Flash (PostgreSQL) split against an InfiniSynapse agent.

Flow per instance (mirrors Spider2's ``run.py`` run_one):

1. Resolve the matching InfiniSynapse PostgreSQL data source for ``db_id``.
2. Submit a ``newTask`` scoped via ``databaseIds`` (+ optional ``engineId``)
   with a debugging prompt built from ``query`` + ``issue_sql``.
3. ``wait_for_task`` until the agent hands control back.
4. Download the workspace zip and extract it under ``output/<instance_id>/``.
5. Harvest ``<instance_id>.sql`` into ``output/pred/flash.jsonl`` as
   ``{"instance_id": ..., "pred_sqls": [...]}``.

The pred jsonl can then be merged for evaluation:

    python ../evaluation/data/merge_pred_into_gt.py \
        --base ../evaluation/data/flash.jsonl \
        --pred output/pred/flash.jsonl \
        --out ../evaluation/data/flash_pred.jsonl
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import queue
import shutil
import sys
import threading
import uuid
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from bird_agent_infini.api.database import (
    download_task_zip,
    list_available_engines,
    new_task,
    select_databases_by_postgres_db_id,
    wait_for_task,
)
from bird_agent_infini.harvest import harvest_workspace, load_pred_ids, upsert_pred
from bird_agent_infini.prompt import build_prompt

#  Logger Configs {{{ #
logger = logging.getLogger("bird_agent_infini")
logger.setLevel(logging.DEBUG)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

_PROJECT_ROOT = Path(__file__).resolve().parent
_LOG_DIR = _PROJECT_ROOT / "logs"
_LOG_DIR.mkdir(exist_ok=True)

file_handler = logging.FileHandler(_LOG_DIR / f"normal-{datetime_str}.log", encoding="utf-8")
debug_handler = logging.FileHandler(_LOG_DIR / f"debug-{datetime_str}.log", encoding="utf-8")
stdout_handler = logging.StreamHandler(sys.stdout)

file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(threadName)s\x1b[1;33m] \x1b[0m%(message)s"
)
file_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(debug_handler)
logger.addHandler(stdout_handler)
#  }}} Logger Configs #

# Repo layout: <repo>/infini/run_flash.py
_REPO_ROOT = _PROJECT_ROOT.parent
DEFAULT_JSONL = _REPO_ROOT / "evaluation" / "data" / "flash.jsonl"
OUTPUT_DIR = _PROJECT_ROOT / "output"
PRED_PATH = OUTPUT_DIR / "pred" / "flash.jsonl"

# Hard timeout for a single InfiniSynapse task run (seconds).
TASK_MAX_WAIT = 1800.0


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the BIRD-Critic Flash split against an InfiniSynapse agent"
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default=str(DEFAULT_JSONL),
        help=f"input GT jsonl with query/issue_sql/db_id (default: {DEFAULT_JSONL})",
    )
    parser.add_argument(
        "--instance_id",
        type=str,
        default=None,
        help="only run the given instance_id(s), comma-separated (e.g. '0' or '0,3,17'). "
             "Order is preserved as given on the command line.",
    )
    parser.add_argument(
        "--range",
        dest="index_range",
        type=str,
        default=None,
        help="run a 1-indexed inclusive range of jsonl lines, formatted as "
             "'start,end' (e.g. '1,10').",
    )
    parser.add_argument(
        "--db_id",
        type=str,
        default=None,
        help="only run tasks for the given db_id(s), comma-separated "
             "(e.g. 'financial,card_games'). Combinable with --instance_id/--range.",
    )
    parser.add_argument(
        "--rerun",
        "--force",
        dest="rerun",
        action="store_true",
        help="re-run instances even if they already have a pred_sqls entry "
             "in output/pred/flash.jsonl.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default=None,
        help="only use the given InfiniSQL engine(s) by name, comma-separated. "
             "Omit to use all available engines (one worker per engine).",
    )
    return parser.parse_args()


def _parse_range(spec: str, total: int) -> tuple[int, int]:
    """Parse a '<start>,<end>' 1-indexed inclusive range string."""
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"--range must look like 'start,end' (got {spec!r})")
    try:
        start, end = int(parts[0]), int(parts[1])
    except ValueError as e:
        raise ValueError(f"--range bounds must be integers (got {spec!r})") from e
    if start < 1 or end < 1:
        raise ValueError(f"--range bounds must be >= 1 (got {spec!r})")
    if start > end:
        raise ValueError(f"--range start must be <= end (got start={start}, end={end})")
    if start > total:
        raise ValueError(f"--range start ({start}) is past the end of the jsonl ({total} lines)")
    return start, min(end, total)


def _resolve_engine_ids(engines: list[dict], engine_spec: str | None) -> list[str]:
    """Return engine ids to use as workers.

    If *engine_spec* is None, use all available engines. Otherwise it is a
    comma-separated list of engine **names** to select.
    """
    available = [item for item in engines if isinstance(item, dict) and item.get("id")]
    if not available:
        return []

    if not engine_spec:
        return [str(item["id"]) for item in available]

    requested = [tok.strip() for tok in engine_spec.split(",") if tok.strip()]
    if not requested:
        raise ValueError(f"--engine is empty after parsing {engine_spec!r}")

    by_name = {
        str(item.get("name") or ""): str(item["id"])
        for item in available
        if item.get("name")
    }

    engine_ids: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()
    for name in requested:
        eid = by_name.get(name)
        if eid is None:
            missing.append(name)
            continue
        if eid not in seen:
            seen.add(eid)
            engine_ids.append(eid)

    if missing:
        known = [(str(item.get("name") or ""), str(item["id"])) for item in available]
        raise ValueError(f"engine name(s) not found: {missing}. Available engines: {known}")
    return engine_ids


def _extract_zip(zip_path: str | os.PathLike, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)


def run_one(task: dict, *, engine_id: str | None = None) -> bool:
    """Run a single Flash instance end-to-end. Returns True on success."""
    instance_id = task["instance_id"]
    db_id = str(task.get("db_id") or "")
    query = task.get("query") or ""
    issue_sql = task.get("issue_sql")

    logger.info("=== Running %s (db_id=%s, engine=%s) ===",
                instance_id, db_id, engine_id or "(default)")

    prompt = build_prompt(instance_id, query, issue_sql)
    task_id = str(uuid.uuid4())

    # 1) Resolve this db_id's PostgreSQL source id and scope the task to it.
    try:
        matching = select_databases_by_postgres_db_id(db_id)
    except Exception as e:
        logger.error("[fail ] %s: failed to resolve postgres source for db_id=%s: %s",
                     instance_id, db_id, e)
        return False

    database_ids = [m["id"] for m in matching if isinstance(m, dict) and m.get("id")]
    if not database_ids:
        logger.error(
            "[fail ] %s: no InfiniSynapse PostgreSQL source matches db_id=%s; "
            "register a postgres source named %r first.",
            instance_id, db_id, db_id,
        )
        return False

    source_names = [m.get("name") for m in matching if isinstance(m, dict)]
    logger.info("[src  ] %s: using postgres source %s (ids=%s)",
                instance_id, source_names, database_ids)

    # 2) Submit the new task scoped to the resolved data source
    try:
        new_task(
            text=prompt,
            task_id=task_id,
            database_ids=database_ids,
            engine_id=engine_id,
        )
    except Exception as e:
        logger.error("[fail ] %s: newTask failed: %s", instance_id, e)
        return False

    logger.info("[task ] %s -> taskId=%s (submitted)", instance_id, task_id)

    # 3) Wait until the runtime actually finished before downloading
    try:
        wait_for_task(
            task_id,
            poll_interval=3.0,
            max_wait=TASK_MAX_WAIT,
            terminal_on_any_ask=True,
            timeout=30.0,
        )
    except TimeoutError as e:
        logger.error("[fail ] %s: task wait timed out: %s", instance_id, e)
        return False
    except Exception as e:
        logger.error("[fail ] %s: wait_for_task error: %s", instance_id, e)
        return False

    # 4) Download workspace zip and extract
    task_output_dir = OUTPUT_DIR / str(instance_id)
    task_output_dir.mkdir(parents=True, exist_ok=True)
    try:
        zip_path = download_task_zip(task_id, task_output_dir)
        logger.info("[zip  ] %s: downloaded %s", instance_id, zip_path)
    except Exception as e:
        logger.error("[fail ] %s: download zip failed: %s", instance_id, e)
        return False

    extract_dir = task_output_dir / "workspace"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    try:
        _extract_zip(zip_path, extract_dir)
    except Exception as e:
        logger.error("[fail ] %s: unzip failed: %s", instance_id, e)
        return False

    # 5) Harvest the deliverable into the pred jsonl
    pred_sqls = harvest_workspace(instance_id, extract_dir)
    if not pred_sqls:
        return False

    upsert_pred(PRED_PATH, instance_id, pred_sqls)
    logger.info("[pred ] %s: saved %d statement(s) -> %s",
                instance_id, len(pred_sqls), PRED_PATH)
    return True


def _run_task_batch(
    tasks: list[dict],
    *,
    engine_ids: list[str],
) -> tuple[int, int]:
    """Run tasks on a global queue with one fixed engine per worker."""
    total = len(tasks)
    if total == 0:
        return 0, 0

    if not engine_ids:
        logger.error("no engine ids available; skipping %d task(s)", total)
        return 0, total

    workers = len(engine_ids)

    def _run_one(idx: int, task: dict, engine_id: str) -> tuple[int, str, bool]:
        instance_id = task.get("instance_id", f"<index-{idx}>")
        db_id = str(task.get("db_id") or "unknown")
        logger.info("---- [%s %d/%d] %s start (engine=%s) ----",
                    db_id, idx, total, instance_id, engine_id)
        try:
            ok = run_one(task, engine_id=engine_id)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.exception("[fail ] %s: unhandled exception: %s", instance_id, e)
            ok = False
        logger.info("---- [%s %d/%d] %s done (ok=%s, engine=%s) ----",
                    db_id, idx, total, instance_id, ok, engine_id)
        return idx, str(instance_id), ok

    if workers <= 1:
        engine_id = engine_ids[0]
        n_ok = 0
        for idx, task in enumerate(tasks, 1):
            try:
                _, _, ok = _run_one(idx, task, engine_id)
            except KeyboardInterrupt:
                logger.warning("Interrupted by user")
                raise
            n_ok += int(ok)
        return n_ok, total

    logger.info("Running %d task(s) with %d engine worker(s): %s",
                total, workers, engine_ids)

    work_q: queue.Queue[tuple[int, dict] | None] = queue.Queue()
    for idx, task in enumerate(tasks, 1):
        work_q.put((idx, task))
    for _ in engine_ids:
        work_q.put(None)

    results: list[tuple[int, str, bool]] = []
    results_lock = threading.Lock()

    def _engine_worker(engine_id: str) -> None:
        while True:
            item = work_q.get()
            try:
                if item is None:
                    return
                idx, task = item
                result = _run_one(idx, task, engine_id)
                with results_lock:
                    results.append(result)
            finally:
                work_q.task_done()

    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="task")
    futures: list[Future] = []
    try:
        for engine_id in engine_ids:
            futures.append(executor.submit(_engine_worker, engine_id))
        for fut in futures:
            fut.result()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user; shutting down engine workers...")
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    n_ok = sum(int(ok) for _, _, ok in results)
    return n_ok, total


def run() -> None:
    args = config()
    logger.info("Args: %s", args)

    jsonl_path = Path(args.jsonl)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        task_configs = [json.loads(line) for line in f if line.strip()]

    if args.instance_id and args.index_range:
        logger.error("--instance_id and --range are mutually exclusive")
        return

    if args.instance_id:
        requested_ids = [tok.strip() for tok in args.instance_id.split(",") if tok.strip()]
        if not requested_ids:
            logger.error("--instance_id is empty after parsing %r", args.instance_id)
            return

        seen: set[str] = set()
        unique_requested: list[str] = []
        for iid in requested_ids:
            if iid in seen:
                logger.warning("[arg  ] duplicate instance_id %r ignored", iid)
                continue
            seen.add(iid)
            unique_requested.append(iid)

        # Flash instance_id is an int in the jsonl; compare as strings.
        by_id = {str(t.get("instance_id")): t for t in task_configs}
        missing = [iid for iid in unique_requested if iid not in by_id]
        if missing:
            logger.error("instance_id(s) %s not found in %s", missing, jsonl_path)
            return

        task_configs = [by_id[iid] for iid in unique_requested]
        logger.info("Running %d explicitly-requested instance(s): %s",
                    len(task_configs), unique_requested)
    elif args.index_range:
        try:
            start, end = _parse_range(args.index_range, len(task_configs))
        except ValueError as e:
            logger.error("%s", e)
            return
        task_configs = task_configs[start - 1:end]
        logger.info("Running jsonl lines %d-%d (%d task(s))", start, end, len(task_configs))

    if args.db_id:
        requested_dbs = [tok.strip() for tok in args.db_id.split(",") if tok.strip()]
        if not requested_dbs:
            logger.error("--db_id is empty after parsing %r", args.db_id)
            return
        db_set = set(requested_dbs)
        before = len(task_configs)
        task_configs = [t for t in task_configs if str(t.get("db_id")) in db_set]
        missing_dbs = sorted(db_set - {str(t.get("db_id")) for t in task_configs})
        if missing_dbs:
            logger.warning("[arg  ] db_id(s) with no matching tasks: %s", missing_dbs)
        if not task_configs:
            logger.error("no tasks left after --db_id filter %s (%d skipped)",
                         requested_dbs, before)
            return
        logger.info("Filtered to %d instance(s) for db_id(s) %s (%d skipped)",
                    len(task_configs), requested_dbs, before - len(task_configs))

    # Skip instances that already have a prediction, unless --rerun.
    if not args.rerun:
        done_ids = load_pred_ids(PRED_PATH)
        if done_ids:
            before = len(task_configs)
            task_configs = [t for t in task_configs if t.get("instance_id") not in done_ids]
            skipped = before - len(task_configs)
            if skipped:
                logger.info("[skip ] %d instance(s) already have pred_sqls in %s "
                            "(use --rerun to redo them)", skipped, PRED_PATH)
        if not task_configs:
            logger.info("Nothing to run: all requested instances already predicted.")
            return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        engines = list_available_engines()
    except Exception as e:
        logger.error("Failed to list available engines: %s", e)
        return

    try:
        engine_ids = _resolve_engine_ids(engines, args.engine)
    except ValueError as e:
        logger.error("%s", e)
        return

    if not engine_ids:
        logger.error(
            "No available InfiniSQL engines found via GET /api/ai_byzer/available; "
            "add at least one enabled engine before running."
        )
        return

    id_to_name = {
        str(item["id"]): str(item.get("name") or item["id"])
        for item in engines
        if isinstance(item, dict) and item.get("id")
    }
    logger.info("Using %d engine worker(s): %s",
                len(engine_ids),
                [(id_to_name.get(eid, eid), eid) for eid in engine_ids])

    n_ok, total = _run_task_batch(task_configs, engine_ids=engine_ids)

    logger.info("All tasks finished: %d/%d succeeded", n_ok, total)
    logger.info("Predictions: %s", PRED_PATH)
    logger.info(
        "Merge for evaluation:\n"
        "  python %s --base %s --pred %s --out %s",
        _REPO_ROOT / "evaluation" / "data" / "merge_pred_into_gt.py",
        DEFAULT_JSONL,
        PRED_PATH,
        _REPO_ROOT / "evaluation" / "data" / "flash_pred.jsonl",
    )


if __name__ == "__main__":
    run()

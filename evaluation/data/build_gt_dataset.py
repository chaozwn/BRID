#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build full GT evaluation JSONL: public HF data + sol_sql/test_cases + schema.

This is step 1 (ground-truth dataset). For agent scoring, next use merge_pred_into_gt.py.
"""

import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
SOL_DIR = ROOT / "sol_raw"
SCHEMA_DIR = ROOT.parent.parent / "baseline" / "data"
OUT_DIR = ROOT
HF_CACHE = ROOT / "hf_public"


DATASETS = {
    "flash": {
        "repo": "birdsql/bird-critic-1.0-flash-exp",
        "file": "data/flash-00000-of-00001.jsonl",
        "sol": "flash_sol.jsonl",
        "schema": "flash_schema.jsonl",
        "out": "flash.jsonl",
    },
    "pg": {
        "repo": "birdsql/bird-critic-1.0-postgresql",
        "file": "data/pg-00000-of-00001.jsonl",
        "sol": "pg_sol.jsonl",
        "schema": "post_schema.jsonl",
        "out": "postgresql_530.jsonl",
    },
    "sqlite": {
        "repo": "birdsql/bird-critic-1.0-sqlite",
        "file": "sqlite-00000-of-00001.jsonl",
        "sol": "sqlite_sol.jsonl",
        "schema": "sqlite_schema.jsonl",
        "out": "sqlite_500.jsonl",
    },
    "open": {
        "repo": "birdsql/bird-critic-1.0-open",
        "file": "data/open-00000-of-000001.jsonl",
        "sol": "open_sol.jsonl",
        "schema": "open_schema.jsonl",
        "out": "open.jsonl",
    },
}


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def dump_jsonl(data_list, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for obj in data_list:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Wrote {len(data_list)} records -> {out_path}")


def index_by_id(records, key="instance_id"):
    return {rec[key]: rec for rec in records}


def merge_records(base_list, *extra_maps):
    merged = []
    for rec in base_list:
        iid = rec["instance_id"]
        out = dict(rec)
        for m in extra_maps:
            if iid in m:
                out.update(m[iid])
        merged.append(out)
    return merged


def download_public(cfg):
    os.makedirs(HF_CACHE, exist_ok=True)
    path = hf_hub_download(
        repo_id=cfg["repo"],
        filename=cfg["file"],
        repo_type="dataset",
        local_dir=str(HF_CACHE / cfg["repo"].split("/")[-1]),
        local_dir_use_symlinks=False,
    )
    return path


def prepare_one(name, cfg):
    print(f"\n=== {name} ===")
    public_path = download_public(cfg)
    base = load_jsonl(public_path)
    sol = index_by_id(load_jsonl(SOL_DIR / cfg["sol"]))
    schema_path = SCHEMA_DIR / cfg["schema"]
    schema = index_by_id(load_jsonl(schema_path)) if schema_path.exists() else {}
    merged = merge_records(base, schema, sol)
    out_path = OUT_DIR / cfg["out"]
    dump_jsonl(merged, out_path)

    # coverage stats
    n_sol = sum(1 for r in merged if r.get("sol_sql"))
    n_tc = sum(1 for r in merged if r.get("test_cases"))
    print(f"  public={len(base)}, sol_map={len(sol)}, with_sol={n_sol}, with_test_cases={n_tc}")
    if n_sol != len(base):
        missing = [r["instance_id"] for r in merged if not r.get("sol_sql")]
        print(f"  [WARN] missing sol_sql for {len(missing)} ids, e.g. {missing[:5]}")
    return merged


def split_open(merged):
    config = {
        "MySQL": ("mysql_100.jsonl", 100),
        "SQLServer": ("mssql_100.jsonl", 100),
        "Oracle": ("oracle_100.jsonl", 100),
        "PostgreSQL": ("postgresql_300.jsonl", 300),
    }
    for dialect, (fname, limit) in config.items():
        subset = [r for r in merged if r.get("dialect") == dialect]
        if len(subset) < limit:
            print(f"Warning: {dialect} has {len(subset)} < expected {limit}, writing all")
            out = subset
        else:
            out = subset[:limit]
        dump_jsonl(out, OUT_DIR / fname)


def validate(path, expect_fields=("db_id", "issue_sql", "sol_sql", "test_cases")):
    data = load_jsonl(path)
    if not data:
        print(f"[WARN] empty: {path}")
        return
    missing = [f for f in expect_fields if f not in data[0]]
    n_sol = sum(1 for r in data if r.get("sol_sql"))
    n_tc = sum(1 for r in data if r.get("test_cases"))
    print(
        f"validate {path.name}: n={len(data)}, sol_sql={n_sol}, test_cases={n_tc}, "
        f"missing_in_first={missing}"
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in ("flash", "pg", "sqlite", "open"):
        merged = prepare_one(name, DATASETS[name])
        if name == "open":
            split_open(merged)

    print("\n=== validation ===")
    for name in (
        "flash.jsonl",
        "postgresql_530.jsonl",
        "sqlite_500.jsonl",
        "open.jsonl",
        "mysql_100.jsonl",
        "mssql_100.jsonl",
        "oracle_100.jsonl",
        "postgresql_300.jsonl",
    ):
        p = OUT_DIR / name
        if p.exists():
            validate(p)


if __name__ == "__main__":
    main()

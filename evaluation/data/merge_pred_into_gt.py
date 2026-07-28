#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge agent pred_sqls into a GT JSONL built by build_gt_dataset.py (by instance_id).

This is step 2 (attach predictions for scoring). Input --base should be GT files like flash.jsonl.
"""

import argparse
import json
import sys
from pathlib import Path


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path}:{i}: {e}") from e
    return data


def dump_jsonl(data_list, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for obj in data_list:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def normalize_pred_sqls(value):
    """Ensure pred_sqls is a list of SQL strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        out = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                if item.strip():
                    out.append(item)
            else:
                out.append(str(item))
        return out
    return [str(value)]


def merge(base_list, pred_map, require_all=False):
    merged = []
    hit = 0
    for rec in base_list:
        iid = rec.get("instance_id")
        out = dict(rec)
        if iid in pred_map:
            pred_rec = pred_map[iid]
            if "pred_sqls" in pred_rec:
                out["pred_sqls"] = normalize_pred_sqls(pred_rec["pred_sqls"])
            else:
                # allow pred file rows that only contain response-extracted fields
                out["pred_sqls"] = []
            hit += 1
        elif require_all:
            raise KeyError(f"Missing prediction for instance_id={iid!r}")
        else:
            out["pred_sqls"] = out.get("pred_sqls", [])
        merged.append(out)
    return merged, hit


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge agent pred_sqls into GT evaluation JSONL by instance_id "
            "(use after build_gt_dataset.py)."
        )
    )
    parser.add_argument(
        "--base",
        required=True,
        help="Full evaluation JSONL (with sol_sql/test_cases), e.g. flash.jsonl",
    )
    parser.add_argument(
        "--pred",
        required=True,
        help="Prediction JSONL with instance_id + pred_sqls, e.g. example_output/.../flash.jsonl",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output merged JSONL path, e.g. flash_pred.jsonl",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail if any base instance_id is missing from pred file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_list = load_jsonl(args.base)
    pred_list = load_jsonl(args.pred)

    pred_map = {}
    dup = 0
    for rec in pred_list:
        if "instance_id" not in rec:
            print(f"[WARN] skip pred row without instance_id: keys={list(rec.keys())}", file=sys.stderr)
            continue
        iid = rec["instance_id"]
        if iid in pred_map:
            dup += 1
        pred_map[iid] = rec

    merged, hit = merge(base_list, pred_map, require_all=args.require_all)
    dump_jsonl(merged, args.out)

    missing = len(base_list) - hit
    unused = len(pred_map) - hit
    print(
        f"Wrote {len(merged)} records -> {args.out}\n"
        f"  base={len(base_list)}, pred={len(pred_list)}, "
        f"matched={hit}, missing_pred={missing}, unused_pred={unused}, dup_pred_ids={dup}"
    )
    if missing:
        miss_ids = [r["instance_id"] for r in merged if not r.get("pred_sqls")]
        # only show those truly without match (empty could also be empty pred)
        truly_missing = [r["instance_id"] for r in base_list if r.get("instance_id") not in pred_map]
        if truly_missing:
            print(f"  missing instance_ids (first 10): {truly_missing[:10]}")


if __name__ == "__main__":
    main()

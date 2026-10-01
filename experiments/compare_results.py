"""Compare regenerated results against the shipped reference results.

For every file in the reference directory that has been regenerated, CSVs are
compared cell by cell (numbers within a relative tolerance, everything else
exactly), JSON files value by value, and NPZ archives array by array.
Wall-clock fields (any column or key containing ``wall``, ``second``,
``latency`` or ``clock``, or ending in ``_s``) depend on the machine and its
load; they are listed but never counted as differences.  Reference files that
were not regenerated (e.g. the live-LLM steps) are listed as such.

Usage:
    .venv/bin/python experiments/compare_results.py
    .venv/bin/python experiments/compare_results.py --rtol 1e-6 --verbose

Exit status is non-zero if any non-timing value differs.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
TIMING = re.compile(r"wall|second|latency|clock|_s$")


def is_timing(name: str) -> bool:
    return bool(TIMING.search(str(name)))


def close(a, b, rtol: float, atol: float) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
    return a == b


def compare_csv(new: pathlib.Path, ref: pathlib.Path, rtol: float,
                atol: float) -> tuple[list[str], list[str]]:
    a, b = pd.read_csv(new), pd.read_csv(ref)
    diffs, timing = [], []
    if list(a.columns) != list(b.columns):
        missing = sorted(set(b.columns) - set(a.columns))
        extra = sorted(set(a.columns) - set(b.columns))
        diffs.append(f"columns differ (missing {missing}, extra {extra})")
    if len(a) != len(b):
        diffs.append(f"{len(a)} rows vs. {len(b)} in the reference")
        return diffs, timing
    for col in [c for c in b.columns if c in a.columns]:
        x, y = a[col], b[col]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            ok = np.isclose(xv, yv, rtol=rtol, atol=atol, equal_nan=True)
        else:
            ok = ((x.isna() & y.isna())
                  | (x.map(str) == y.map(str))).to_numpy(bool)
        n = int((~ok).sum())
        if n:
            i = int(np.flatnonzero(~ok)[0])
            msg = f"{col}: {n}/{len(ok)} cells (row {i}: {x.iloc[i]!r} vs. {y.iloc[i]!r})"
            (timing if is_timing(col) else diffs).append(msg)
    return diffs, timing


def compare_json(new: pathlib.Path, ref: pathlib.Path, rtol: float,
                 atol: float) -> tuple[list[str], list[str]]:
    diffs, timing = [], []

    def walk(a, b, path: str, in_timing: bool) -> None:
        if isinstance(b, dict) and isinstance(a, dict):
            for k in sorted(set(a) | set(b)):
                p = f"{path}.{k}" if path else str(k)
                t = in_timing or is_timing(k)
                if k not in a or k not in b:
                    (timing if t else diffs).append(f"{p}: only in "
                                                    f"{'reference' if k in b else 'new'}")
                else:
                    walk(a[k], b[k], p, t)
        elif isinstance(b, list) and isinstance(a, list):
            if len(a) != len(b):
                (timing if in_timing else diffs).append(
                    f"{path}: length {len(a)} vs. {len(b)}")
                return
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]", in_timing)
        elif not close(a, b, rtol, atol):
            (timing if in_timing else diffs).append(f"{path}: {a!r} vs. {b!r}")

    walk(json.loads(new.read_text()), json.loads(ref.read_text()), "", False)
    return diffs, timing


def compare_npz(new: pathlib.Path, ref: pathlib.Path, rtol: float,
                atol: float) -> tuple[list[str], list[str]]:
    a = np.load(new, allow_pickle=True)
    b = np.load(ref, allow_pickle=True)
    diffs = []
    if sorted(a.files) != sorted(b.files):
        diffs.append(f"arrays differ: {sorted(a.files)} vs. {sorted(b.files)}")
    for k in sorted(set(a.files) & set(b.files)):
        x, y = a[k], b[k]
        if x.shape != y.shape:
            diffs.append(f"{k}: shape {x.shape} vs. {y.shape}")
        elif x.dtype.kind in "fc":
            if not np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True):
                diffs.append(f"{k}: values differ")
        elif x.dtype.kind == "O":
            if any(not np.array_equal(np.asarray(u), np.asarray(v))
                   for u, v in zip(x.ravel(), y.ravel())):
                diffs.append(f"{k}: values differ")
        elif not np.array_equal(x, y):
            diffs.append(f"{k}: values differ")
    return diffs, []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--reference", default=str(ROOT / "reference_results"))
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-9)
    ap.add_argument("--verbose", action="store_true",
                    help="also list the timing fields that differ")
    args = ap.parse_args()

    res, ref = pathlib.Path(args.results), pathlib.Path(args.reference)
    compare = {".csv": compare_csv, ".json": compare_json, ".npz": compare_npz}
    match, differ, absent = [], [], []
    for r in sorted(ref.iterdir()):
        if r.suffix not in compare:
            continue
        n = res / r.name
        if not n.exists():
            absent.append(r.name)
            continue
        try:
            diffs, timing = compare[r.suffix](n, r, args.rtol, args.atol)
        except Exception as e:                    # unreadable, malformed, ...
            diffs, timing = [f"could not compare: {e}"], []
        if diffs:
            differ.append(r.name)
            print(f"DIFF   {r.name}")
            for d in diffs[:10]:
                print(f"         {d}")
            if len(diffs) > 10:
                print(f"         ... and {len(diffs) - 10} more")
        else:
            match.append(r.name)
            note = f"  ({len(timing)} timing fields differ)" if timing else ""
            print(f"MATCH  {r.name}{note}")
        if args.verbose:
            for t in timing:
                print(f"         [timing] {t}")

    if absent:
        print(f"\nnot regenerated ({len(absent)}):")
        for a in absent:
            print(f"         {a}")
    print(f"\n{len(match)} match, {len(differ)} differ, "
          f"{len(absent)} not regenerated")
    sys.exit(1 if differ else 0)


if __name__ == "__main__":
    main()

"""Build the report folder for a training session.

    python3 report_run.py runs/pls_variety_reflectance_both_allh_d0.json
    python3 report_run.py --latest                # most recent run
    python3 report_run.py --all                   # every run in runs/
    python3 report_run.py --dataset               # the dataset overview
    python3 report_run.py --latest --mode dark    # dark-surface figures

Reads the run's JSON and its .npz of per-view predictions; touches no model and
retrains nothing, so a figure can be added or restyled at any time and every
report regenerated in seconds.

train_pls.py and train_cnn.py call this automatically at the end of a run
unless given --no-report.
"""
import argparse
import sys

import config
from barley import runlog
from reporting import session


def latest():
    runs = sorted(config.RUNS.glob("*.json"), key=lambda p: p.stat().st_mtime)
    runs = [p for p in runs if not p.name.endswith("_sweep.json")]
    if not runs:
        raise SystemExit(f"no runs in {config.RUNS} -- train something first")
    return runs[-1]


def one(path, mode):
    """-> the report dir, or None. Never raises: under --all, one unreadable
    run used to abort the loop, so every later report silently never got built.
    main() still exits non-zero if anything failed."""
    try:
        return _one(path, mode)
    except Exception as e:
        print(f"  FAILED {path}: {type(e).__name__}: {e}")
        return False


def _one(path, mode):
    meta, pred, name = runlog.load(path)
    if "task" not in meta:
        print(f"  skipping {name}: not a single-run file")
        return None
    out, made = session.build(meta, pred, name, mode=mode)
    print(f"{name}\n  -> {out}/summary.md")
    for m in made:
        print(f"     {m.name}")
    if not pred:
        print("     (no .npz -- per-dish and spectra figures need a re-run)")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?", help="path to runs/<name>.json, or the bare name")
    p.add_argument("--latest", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--dataset", action="store_true", help="dataset overview instead")
    p.add_argument("--mode", choices=["light", "dark"], default="light")
    args = p.parse_args()

    if args.dataset:
        out, made = session.dataset_report(mode=args.mode)
        print(f"dataset\n  -> {out}/summary.md")
        for m in made:
            print(f"     {m.name}")
        return 0

    if args.all:
        failed = 0
        for path in sorted(config.RUNS.glob("*.json")):
            if path.name.endswith("_sweep.json"):
                continue
            if one(path, args.mode) is False:
                failed += 1
        if failed:
            print(f"\n{failed} run(s) failed to report")
        return 1 if failed else 0

    path = args.run or (latest() if args.latest else None)
    if path is None:
        p.error("give a run path, or --latest / --all / --dataset")
    return 1 if one(path, args.mode) is False else 0


if __name__ == "__main__":
    sys.exit(main())

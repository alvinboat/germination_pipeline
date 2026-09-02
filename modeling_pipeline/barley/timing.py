"""The measured clock: when each capture happened, and when each plate was scored.

`config.HOURS` and `config.GERMINATION_DAYS` are the NOMINAL plan -- 0/8/24 h
for the captures, 24/48/72/96/120 h for the scoring. `timing.json` records what
actually happened; this module reads it.

That file is a hand-acquired INPUT, not a build product. Capture times came from
file mtimes on the acquisition machine and visit times from EXIF on the scoring
photos; the tooling that collected them has been removed, so it cannot be
rebuilt from anything here. It is versioned for that reason. Do not delete it.

    from barley import timing
    T = timing.load()
    timing.capture_hours("reflectance", 9, 7, "dorsal")   # 7.42, not 8.0
    timing.visit_hours(7)                                 # [22.9, 48.6, ...]
    lo, hi = timing.intervals(uids, dishes, table)         # germination bracket

EVERY CLOCK IS PER DISH
The day-1 session took 8.3 h to work through 25 plates, so there is no single
t=0. Each dish's zero is its own dry scan, and its later scans and photos are
measured from that. A global zero would have to carry that 8.3 h as label noise.

LABELS ARE INTERVALS
"Day 2" means the kernel was ungerminated in the day-1 photo and germinated in
the day-2 photo. That is a bracket, not a time -- and the brackets are 21.4 to
28.2 h wide, not a tidy 24. A never-germinator is censored at its own dish's
last visit, which lands anywhere from 120.9 to 128.6 h, not at 120.

Nothing here overrides `config`. The nominal values stay where they are and stay
correct as a description of the protocol; this is what the protocol produced.
"""
import datetime as dt
import json

import numpy as np

import config

_CACHE = None
TIMING_JSON = config.TIMING_JSON


def available():
    return TIMING_JSON.exists()


def load(path=None):
    """The parsed timing.json, cached."""
    global _CACHE
    if _CACHE is None or path is not None:
        p = path or TIMING_JSON
        if not p.exists():
            raise SystemExit(
                f"no {p}. This file is a hand-acquired input, not a build "
                "product -- it cannot be regenerated here. Restore it from "
                "version control.")
        data = json.loads(p.read_text())
        if path is None:
            _CACHE = data
        else:
            return data
    return _CACHE


def t0(dish):
    """The dish's zero: its own day1 dry scan, immediately before wetting."""
    return dt.datetime.fromisoformat(load()["t0_by_dish"][str(int(dish))])


# The three lookups below have no caller in this repo, and that is deliberate.
# `attach()` builds the same columns in bulk for Index.load, and the
# germination-TIME models that consumed `intervals()` were deleted -- the
# measured brackets they encode are the thing that survived that deletion.
# They are the query API for ad-hoc work and for any future timing model.
def capture_hours(mode, day, dish, side):
    """Measured hours between the dish's t0 and this capture. NaN if unknown.

    Check `capture_time_source` on the index before quoting one of these: a
    transmittance view may carry the reflectance timestamp for the same capture
    (`proxy_reflectance`, bounded by one dish slot, ~14 min) rather than a
    measured one. It is never replaced with the NOMINAL value, because the whole
    point of this module is that the nominal value is wrong by up to 4 h.
    """
    rec = load()["captures"].get(f"{mode}|{int(day)}|{int(dish)}|{side}")
    return float(rec["hours"]) if rec else float("nan")


def capture_time(mode, day, dish, side):
    rec = load()["captures"].get(f"{mode}|{int(day)}|{int(dish)}|{side}")
    return dt.datetime.fromisoformat(rec["time"]) if rec else None


def visit_hours(dish):
    """-> [hours since t0] for that dish's scoring visits, in order."""
    return [float(v["hours"]) for v in load()["visits_by_dish"][str(int(dish))]]


def censor_hours(dish):
    """When observation stopped for this dish -- its last actual visit."""
    return visit_hours(dish)[-1]


def intervals(kernel_uids, dishes, table):
    """Germination brackets. -> (lower, upper) float arrays, hours since t0.

    `table` is {kernel_uid: first_day or None} from
    `germination.read_germination_labels`. A kernel absent from it is unscored and
    comes back (nan, nan) -- callers must drop those, never read them as never.

    For a kernel first seen germinated at visit k, the event happened after
    visit k-1 and by visit k, so the bracket is (visit[k-2], visit[k-1]] with
    visit 0 being the wetting moment at hour 0. A never-germinator gets
    (last visit, inf): right-censored, fully observed, not missing.
    """
    lo = np.full(len(kernel_uids), np.nan)
    hi = np.full(len(kernel_uids), np.nan)
    for i, (u, d) in enumerate(zip(kernel_uids, dishes)):
        u = str(u)
        if u not in table:
            continue
        vh = visit_hours(int(d))
        day = table[u]
        if day is None:
            lo[i], hi[i] = vh[-1], np.inf
        else:
            k = int(day)
            lo[i] = 0.0 if k == 1 else vh[k - 2]
            hi[i] = vh[k - 1]
    return lo, hi


def attach(idx, quiet=False):
    """Add `capture_time` and `capture_hours` columns to an Index, in place.

    Called by `Index.load`. Degrades to a warning rather than an error when
    timing.json is missing, so the rest of the pipeline still runs.
    """
    if not available():
        if not quiet:
            print(f"  note: no {TIMING_JSON.name} -- capture_hours unavailable")
        return idx
    n = len(idx)
    times, hrs = np.empty(n, object), np.full(n, np.nan)
    src = np.full(n, "none", object)
    for i in range(n):
        key = (str(idx["mode"][i]), int(idx["day_folder"][i]),
               int(idx["dish"][i]), str(idx["side"][i]))
        rec = load()["captures"].get("%s|%d|%d|%s" % key)
        if rec:
            times[i], hrs[i] = rec["time"], float(rec["hours"])
            src[i] = rec.get("source", "measured")
    idx.cols["capture_time"] = times
    idx.cols["capture_hours"] = hrs
    # Provenance travels with the number. A borrowed timestamp that cannot be
    # told from a measured one is how a proxy quietly becomes a fact.
    idx.cols["capture_time_source"] = src
    if not quiet:
        from collections import Counter
        c = Counter(src.tolist())
        if set(c) - {"measured"}:
            print("  capture times: " + ", ".join(f"{v} {k}" for k, v in
                                                  sorted(c.items())))
    return idx

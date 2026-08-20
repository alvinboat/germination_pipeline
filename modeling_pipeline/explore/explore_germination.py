"""How long the kernels took to germinate -- one histogram overall, one per variety.

    python3 explore/explore_germination.py            # -> exploration/germination/

The question is whether germination time has a shape: one mode or two, a fast
variety and a slow one, a dormant tail that is really a separate population.
Everything downstream depends on the answer. A single sharp mode means the
survival target is nearly binary and the extra columns buy little; two modes
mean there is something to separate and the staircase earns its place.

WHAT IS COUNTED
Kernels, not views. The index holds eight views of each kernel -- two modes,
two faces, two capture times -- and every one of them carries the same
germination label, so a histogram over index rows would multiply every bar by
eight and say nothing new. The unit here is the kernel.

WHY THE BARS SPAN A RANGE
A kernel labelled "day 2" was seen ungerminated at one visit and germinated at
the next. That is an INTERVAL, not a time -- so each bar is drawn across the
interval it represents on a real hour axis, not parked over a tick labelled
"48 h". The bar edges are the mean visit times recorded in `timing.json`;
the caps beneath each edge show how much that visit moved between dishes, which
is up to 5 h. Nominal 24/48/72/96/120 is the protocol, and it is wrong: the
visits actually landed at 22.6 / 48.4 / 72.0 / 95.7 / 123.7 h.

WHY `never` IS DRAWN APART
`never` is not a sixth, later time: it is a right-censored observation, "had not
germinated when scoring stopped". Drawn flush against the last bin it would read
as one more bin, which is the exact misreading `barley.germination` exists to
prevent -- so it sits past an axis break, in a different colour, labelled with
the measured horizon (120.9-128.6 h, not 120).

SCOPE
`config.EXCLUDED_VARIETIES` is applied, so prospect2 -- 110 of 110 never -- is
not here. Its histogram would be a single bar over `never` and its inclusion
would put a fifth of the plate on one side of every comparison below.
"""
import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt              # noqa: E402

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                     # noqa: E402
from barley import index as index_mod, germination, timing   # noqa: E402
from reporting import style as S                  # noqa: E402

EVENT_INK = 0       # categorical slot for a germination-day bar
NEVER_INK = 1       # ... and for the censored bar, which is not a time
GAP = 0.6           # blank x-units between day 5 and `never`


# --------------------------------------------------------------------- data --
def kernel_table(idx, labels):
    """-> [{kernel_uid, dish, variety, day}] one row per kernel, not per view.

    `day` is 1..5, or None for scored-but-never. Kernels the sheet has not
    scored are dropped and counted by the caller: an unscored kernel is a
    missing label, and putting it anywhere on this axis would invent data.
    """
    seen, rows, unscored = set(), [], []
    for u, d, v in zip(idx["kernel_uid"], idx["dish"], idx["variety"]):
        if u in seen:
            continue
        seen.add(u)
        if u not in labels:
            unscored.append(u)
            continue
        rows.append({"kernel_uid": u, "dish": int(d), "variety": int(v),
                     "day": labels[u]})
    return rows, unscored


def visit_edges(rows):
    """-> (mean lower, mean upper, lo-range, hi-range) per bin, in hours.

    Averaged over the dishes actually present in `rows`, so a per-variety figure
    is drawn on its own dishes' clock rather than the whole plate's.
    """
    dishes = sorted({r["dish"] for r in rows})
    V = np.array([timing.visit_hours(d) for d in dishes])      # (n_dish, 5)
    lower = np.concatenate([[0.0], V.mean(0)[:-1]])
    upper = V.mean(0)
    lo_rng = np.concatenate([[(0.0, 0.0)], list(zip(V.min(0)[:-1], V.max(0)[:-1]))])
    hi_rng = np.array(list(zip(V.min(0), V.max(0))))
    return lower, upper, lo_rng, hi_rng, V


def profile(rows):
    """-> the numbers every figure and every table on this page is built from."""
    days = config.GERMINATION_DAYS
    c = Counter(r["day"] for r in rows)
    counts = [c[d] for d in days]
    never = c[None]
    n = len(rows)
    germ = n - never
    # Median over the kernels that germinated. A median "time" computed with the
    # never-germinators folded in at any finite value would be a fabrication;
    # with them at infinity it is defined only while they are under half.
    times = sorted(r["day"] for r in rows if r["day"] is not None)
    return {
        "n": n, "germinated": germ, "never": never,
        "counts": counts,
        "pct": [100 * x / n if n else 0.0 for x in counts],
        "pct_germinated": 100 * germ / n if n else 0.0,
        "median_day": float(np.median(times)) if times else None,
        "mode_day": days[int(np.argmax(counts))] if germ else None,
        # Cumulative incidence: share of ALL kernels up by each visit. It ends
        # at pct_germinated, not at 100 -- the shortfall is the censored group.
        "cum_pct": list(np.cumsum(counts) / n * 100) if n else [],
        # The measured visit hours for the dishes in THIS group, so a
        # per-variety curve is drawn on its own dishes' clock.
        "visit_h": list(np.mean([timing.visit_hours(d)
                                 for d in sorted({r["dish"] for r in rows})], axis=0)),
    }


# ------------------------------------------------------------------ figures --
def histogram(prof, edges, title, subtitle, out, ymax=None):
    """Counts against a real hour axis, each bar spanning its own interval."""
    lower, upper, lo_rng, hi_rng, V = edges
    top = ymax if ymax is not None else max([*prof["counts"], prof["never"], 1])
    width = upper - lower
    never_w = float(width.mean())
    brk = float(hi_rng[-1][1]) + 0.35 * never_w          # axis break sits here
    x_never = brk + 0.45 * never_w

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.bar(lower, prof["counts"], width=width, align="edge",
           color=S.series(EVENT_INK), edgecolor=S.ink("surface"), lw=1.0, zorder=3)
    ax.bar([x_never], [prof["never"]], width=never_w, align="edge",
           color=S.series(NEVER_INK), edgecolor=S.ink("surface"), lw=1.0, zorder=3)

    # A bar edge is the MEAN visit time; the real edge moved by up to 7.6 h
    # between dishes. That was drawn here as a range marker and it cost more
    # attention than it returned -- the same fact is a column in summary.md and
    # the whole point of scoring_visits.png, both of which say it better.
    for xi, w, v in zip([*lower, x_never], [*width, never_w],
                        [*prof["counts"], prof["never"]]):
        if v:
            ax.text(xi + w / 2, v + top * 0.02, f"{v}", ha="center", va="bottom",
                    fontsize=9, color=S.ink("secondary"))

    for x in (brk - 0.1 * never_w, brk + 0.1 * never_w):
        ax.plot([x, x], [0, top * 1.14], color=S.ink("surface"), lw=3.5,
                zorder=5, clip_on=False)
        ax.plot([x - 1.2, x + 1.2], [-top * 0.012, top * 0.012],
                color=S.ink("axis"), lw=1.0, zorder=6, clip_on=False)

    ticks = [0.0, *upper]
    ax.set_xticks(ticks + [x_never + never_w / 2])
    ax.set_xticklabels([f"{t:.0f}" for t in ticks] + ["never"])
    ax.set_xlim(-0.02 * upper[-1], x_never + never_w * 1.15)
    ax.set_ylim(0, top * 1.14)
    ax.set_xlabel("hours since that dish was wetted   —   each bar spans the "
                  "interval the germination happened in")
    ax.set_ylabel("kernels")
    S.hide_grid_x(ax)
    S.title(ax, title, subtitle)
    ax.annotate(f"right-censored\nat {V[:, -1].min():.0f}-{V[:, -1].max():.0f} h",
                xy=(x_never + never_w / 2, top * 0.06), ha="center", va="bottom",
                fontsize=7.5, color=S.on_dark(S.series(NEVER_INK)))
    fig.savefig(out)
    plt.close(fig)


def fig_visits(rows, out):
    """When each dish was actually scored. The reason the bars span a range."""
    dishes = sorted({r["dish"] for r in rows})
    V = np.array([timing.visit_hours(d) for d in dishes])
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    for k in range(V.shape[1]):
        ax.scatter(V[:, k], dishes, s=22, linewidths=0, color=S.series(0), zorder=3)
        ax.annotate(f"visit {k + 1}", xy=(V[:, k].mean(), max(dishes) + 1.4),
                    ha="center", fontsize=8, color=S.ink("secondary"))
        ax.axvline(config.germination_hours()[k], color=S.series(1), lw=1.0,
                   ls=":", zorder=2)
    for d, row in zip(dishes, V):
        ax.plot(row, [d] * len(row), color=S.ink("grid"), lw=1.0, zorder=1)
    ax.set_ylim(min(dishes) - 1.5, max(dishes) + 3)
    ax.set_xlabel("hours since that dish was wetted   —   each bar spans the "
                  "interval the germination happened in")
    ax.set_ylabel("dish")
    S.title(ax, "the scoring visits are not 24 h apart, and not the same for every dish",
            "dotted lines are the nominal 24/48/72/96/120 h; dots are what happened")
    fig.savefig(out)
    plt.close(fig)


def _spread(values, gap):
    """Nudge label positions apart, keeping their order. -> new positions.

    The four curves finish within four points of each other, so direct labels
    at their endpoints overprint. Moving the labels and drawing a leader to the
    real endpoint keeps them readable without moving the data.
    """
    order = np.argsort(values)
    out = np.asarray(values, float).copy()
    for i in range(1, len(order)):
        prev, cur = order[i - 1], order[i]
        out[cur] = max(out[cur], out[prev] + gap)
    return out


def compare(profs, out):
    """Cumulative incidence against the measured visit hours, one line per variety.

    The one place the categorical hues belong: four series over a shared axis
    where position cannot tell them apart. Direct-labelled, as the palette's
    light-mode contrast requires.
    """
    days = config.GERMINATION_DAYS
    hours = np.mean([p["visit_h"] for _, p in profs], axis=0)
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ends = [p["cum_pct"][-1] for _, p in profs]
    label_y = _spread(ends, gap=4.0)
    for i, (name, p) in enumerate(profs):
        ax.plot(hours, p["cum_pct"], color=S.series(i), marker="o", zorder=3)
        ax.plot([hours[-1], hours[-1] + 9], [ends[i], label_y[i]],
                color=S.series(i), lw=0.8, zorder=2)
        ax.annotate(f"{name}  {p['pct_germinated']:.0f}%",
                    xy=(hours[-1] + 10, label_y[i]), va="center", fontsize=8.5,
                    color=S.series(i))
    ax.set_xticks(list(hours))
    ax.set_xticklabels([f"visit {d}\n{h:.0f} h" for d, h in zip(days, hours)])
    ax.set_xlim(hours[0] - 8, hours[-1] + 46)
    ax.set_ylim(0, max(105, float(label_y.max()) + 5))
    ax.set_ylabel("% of kernels germinated by then")
    S.hide_grid_x(ax)
    S.title(ax, "cumulative germination, by variety",
            "each line ends at its germination rate; the gap to 100% is the "
            "never-germinated group")
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------------- tables --
def write_tables(out_dir, overall, per_variety, unscored, n_views):
    days = config.GERMINATION_DAYS
    with (out_dir / "counts.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "n_kernels", *[f"day{d}" for d in days], "never",
                    "germinated", "pct_germinated", "median_day_of_germinated",
                    "modal_day"])
        for name, p in [("all", overall), *per_variety]:
            w.writerow([name, p["n"], *p["counts"], p["never"], p["germinated"],
                        f"{p['pct_germinated']:.1f}",
                        "" if p["median_day"] is None else f"{p['median_day']:.1f}",
                        "" if p["mode_day"] is None else p["mode_day"]])

    def row(name, p):
        return ("| " + " | ".join([name, str(p["n"]), *[str(c) for c in p["counts"]],
                                   str(p["never"]), f"{p['pct_germinated']:.0f}%",
                                   "-" if p["median_day"] is None
                                   else f"{p['median_day']:.1f}"]) + " |")

    edge = [0.0] + list(np.mean([timing.visit_hours(d) for d in sorted(
        {r["dish"] for r in overall["_rows"]})], axis=0))
    head = ("| group | n | "
            + " | ".join(f"{edge[i]:.0f}-{edge[i + 1]:.0f} h" for i in range(len(days)))
            + " | never | germinated | median visit |")
    rule = "|" + "---|" * (len(days) + 5)
    lines = [head, rule, row("**all**", overall)]  # columns are intervals, not days
    lines += [row(n, p) for n, p in per_variety]

    V = np.array([timing.visit_hours(d) for d in sorted(
        {r["dish"] for r in overall["_rows"]})])
    vis = ["| visit | nominal | measured mean | range across dishes |",
           "|---|---|---|---|"]
    for k, nom in enumerate(config.germination_hours()):
        vis.append(f"| {k + 1} | {nom:.0f} h | **{V[:, k].mean():.1f} h** | "
                   f"{V[:, k].min():.1f} - {V[:, k].max():.1f} h |")
    (out_dir / "summary.md").write_text(SUMMARY.format(
        visits="\n".join(vis),
        censor=f"{V[:, -1].min():.1f}-{V[:, -1].max():.1f} h",
        table="\n".join(lines),
        kernels=overall["n"], views=n_views,
        excluded=", ".join(config.excluded_names()) or "none",
        unscored=len(unscored),
        built=time.strftime("%Y-%m-%dT%H:%M:%S")))


SUMMARY = """# germination time

{kernels} kernels ({views} index views), one row per kernel. Excluded varieties:
{excluded}. Unscored kernels dropped: {unscored}. Built {built} by
`python3 explore/explore_germination.py`.

Counts are kernels. Every count is a count of INTERVALS: a kernel in the second
column was seen ungerminated at visit 1 and germinated at visit 2, so all that
is known is that it came up between them. The bars are drawn across those
intervals on a real hour axis rather than parked on a nominal tick.

The visits are not 24 h apart and not the same for every dish -- each plate has
its own clock, starting at its own dry scan. Measured, from `timing.json`:

{visits}

`never` is right-censored at that dish's last visit, which lands at
{censor} -- not at the nominal 120 h. It is not a longer germination time.

{table}

## figures

* `all_varieties.png` -- the pooled histogram.
* `variety<N>_<name>.png` -- one per variety. These four share a y-axis, so the
  bars are comparable between them by eye; the pooled figure does not, and its
  bars are not to the same scale.
* `scoring_visits.png` -- when each dish was actually scored, against the
  nominal 24/48/72/96/120 h. This is why the bars span a range.
* `varieties_compared.png` -- cumulative incidence, all four on one axis. The
  shape question is easiest to read here: a variety that is merely slower has
  the same curve shifted right, while a variety with a distinct dormant
  sub-population flattens early and never catches up.

`counts.csv` is the same numbers, unrounded. Its column names are still
`day1..day5`; they mean the same five intervals.
"""


# --------------------------------------------------------------------- main --
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=config.EXPLORATION / "germination")
    p.add_argument("--include-excluded", action="store_true",
                   help="also plot config.EXCLUDED_VARIETIES")
    args = p.parse_args()

    S.use("light")
    idx = index_mod.Index.load(include_excluded=args.include_excluded)
    labels = germination.read_germination_labels(
        cell_map=germination.build_cell_map())      # cell map over the whole plate
    rows, unscored = kernel_table(idx, labels)
    if unscored:
        print(f"  dropping {len(unscored)} unscored kernel(s): "
              f"{', '.join(sorted(unscored)[:5])}")
    if not rows:
        raise SystemExit("no scored kernels in scope")

    args.out.mkdir(parents=True, exist_ok=True)
    overall = profile(rows)
    overall["_rows"] = rows
    varieties = [v for v in sorted({r["variety"] for r in rows})]
    per_variety = [(config.VARIETY_NAMES[v - 1],
                    profile([r for r in rows if r["variety"] == v])) for v in varieties]

    # One y-limit for the four per-variety figures. Without it every panel
    # rescales to its own tallest bar and four different shapes look alike.
    ymax = max(max([*p["counts"], p["never"]]) for _, p in per_variety)

    edges_all = visit_edges(rows)
    fig_visits(rows, args.out / "scoring_visits.png")
    histogram(overall, edges_all, "germination time, all varieties",
              f"{overall['n']} kernels  ·  {overall['germinated']} germinated "
              f"({overall['pct_germinated']:.0f}%)  ·  modal interval "
              f"{edges_all[0][1]:.0f}-{edges_all[1][1]:.0f} h  ·  "
              f"{overall['never']} never",
              args.out / "all_varieties.png")
    print(f"all           n={overall['n']:3d}  {overall['counts']}  "
          f"never={overall['never']:3d}  {overall['pct_germinated']:.0f}% germinated")

    for v, (name, prof) in zip(varieties, per_variety):
        vrows = [r for r in rows if r["variety"] == v]
        dishes = sorted({r["dish"] for r in vrows})
        histogram(prof, visit_edges(vrows), f"germination time — {name}",
                  f"dishes {dishes[0]}-{dishes[-1]}  ·  {prof['n']} kernels  ·  "
                  f"{prof['germinated']} germinated ({prof['pct_germinated']:.0f}%)"
                  f"  ·  median visit "
                  + ("-" if prof["median_day"] is None else f"{prof['median_day']:.1f}")
                  + f"  ·  {prof['never']} never",
                  args.out / f"variety{v}_{name}.png", ymax=ymax)
        print(f"{name:13s} n={prof['n']:3d}  {prof['counts']}  "
              f"never={prof['never']:3d}  {prof['pct_germinated']:.0f}% germinated")

    compare(per_variety, args.out / "varieties_compared.png")
    write_tables(args.out, overall, per_variety, unscored, len(idx))

    print(f"\n{len(per_variety) + 2} figures -> {args.out}")
    print(f"tables  : {args.out / 'summary.md'}, {args.out / 'counts.csv'}")


if __name__ == "__main__":
    sys.exit(main())

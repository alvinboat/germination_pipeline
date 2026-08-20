# germination time

438 kernels (3502 index views), one row per kernel. Excluded varieties:
prospect2. Unscored kernels dropped: 0. Built 2026-08-20T13:10:21 by
`python3 explore/explore_germination.py`.

Counts are kernels. Every count is a count of INTERVALS: a kernel in the second
column was seen ungerminated at visit 1 and germinated at visit 2, so all that
is known is that it came up between them. The bars are drawn across those
intervals on a real hour axis rather than parked on a nominal tick.

The visits are not 24 h apart and not the same for every dish -- each plate has
its own clock, starting at its own dry scan. Measured, from `timing.json`:

| visit | nominal | measured mean | range across dishes |
|---|---|---|---|
| 1 | 24 h | **22.6 h** | 21.4 - 26.0 h |
| 2 | 48 h | **48.1 h** | 45.4 - 53.4 h |
| 3 | 72 h | **71.8 h** | 69.0 - 77.1 h |
| 4 | 96 h | **95.4 h** | 92.7 - 100.8 h |
| 5 | 120 h | **123.5 h** | 120.9 - 128.6 h |

`never` is right-censored at that dish's last visit, which lands at
120.9-128.6 h -- not at the nominal 120 h. It is not a longer germination time.

| group | n | 0-23 h | 23-48 h | 48-72 h | 72-95 h | 95-123 h | never | germinated | median visit |
|---|---|---|---|---|---|---|---|---|---|
| **all** | 438 | 101 | 236 | 33 | 12 | 8 | 48 | 89% | 2.0 |
| prospect1 | 110 | 36 | 58 | 5 | 0 | 1 | 10 | 91% | 2.0 |
| laureate1 | 109 | 32 | 49 | 7 | 3 | 5 | 13 | 88% | 2.0 |
| laureate2 | 110 | 29 | 55 | 6 | 4 | 2 | 14 | 87% | 2.0 |
| unknown | 109 | 4 | 74 | 15 | 5 | 0 | 11 | 90% | 2.0 |

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

"""Where and when trains turn short of the terminus.

Every trip that doesn't run terminus to terminus is grouped by its pattern
(origin→destination). A pattern that runs at least REGULAR times a day in one
direction, and not only at the edges of the day (before 07:00 or after
22:00), is called a short-turn service. The rest are labelled "positioning":
rare patterns, mostly a train entering service from a depot or siding, or
leaving service to one. The trip chaining in explore_trips (block) shows
which: a trip that starts a block is a train entering service, one that ends
a block is a train leaving it. Line 6 trips that pass stations are labelled
"express" whatever their termini. The thresholds are judgement calls;
patterns.csv has the raw counts so they can be revisited.

Outputs (in --out): patterns.csv, positioning.csv, hourly.csv,
short_turn_timeline.png, short_turn_share.png, coverage_map.png and summary.md.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl

from explore_common import (
    INK, INK_2, PERIOD_ORDER, SERIES, SURFACE, hhmm, line_order, load, md_table, period_of, plt,
    save, time_axis,
)
from explore_map import SubwayMap, segment_values

REGULAR = 6
# Patterns that only run before the first or after the second time are
# trains entering or leaving service, not a short-turn service.
EDGE_OF_DAY = (7 * 60, 22 * 60)


def main(explore_dir, lines_csv, map_json5, closures_csv, out_dir):
    out = Path(out_dir)
    trips, stops = load(explore_dir)
    lines = line_order(lines_csv)
    # Which trips start or end a train's working (see explore_trips.chain_blocks).
    trips["enters_service"] = trips.layover_before_min.isna()
    last = trips.sort_values("start_min").groupby("block").trip_id.last()
    trips["leaves_service"] = trips.trip_id.isin(last)
    all_trips = trips
    trips = trips[trips.kind != "loop"].copy()
    trips["period"] = period_of(trips.start_min)
    per_dir = trips.groupby(["line_id", "direction"]).size().rename("dir_trips")

    g = trips.groupby(["line_id", "direction", "pattern", "kind"], observed=True)
    pat = g.agg(trips=("trip_id", "size"), stops=("n_stops", "median"),
                first_dep=("start_min", "min"), last_dep=("start_min", "max"),
                enter_service=("enters_service", "sum"),
                leave_service=("leaves_service", "sum")).reset_index()
    pat = pat.join(per_dir, on=["line_id", "direction"])
    pat["share"] = pat.trips / pat.dir_trips
    key = ["line_id", "direction", "pattern"]
    peak = trips[trips.period.isin(["am_peak", "pm_peak"])].groupby(key).size().rename("in_peak")
    pat = pat.join(peak, on=key)
    pat["in_peak"] = pat.in_peak.fillna(0).astype(int)
    edge_of_day = (pat.last_dep < EDGE_OF_DAY[0]) | (pat.first_dep >= EDGE_OF_DAY[1])
    pat["role"] = np.select(
        [pat.pattern.str.endswith("express"), pat.kind == "full",
         (pat.trips >= REGULAR) & ~edge_of_day],
        ["express", "terminus to terminus", "short-turn service"],
        "positioning",
    )
    pat["line_rank"] = pat.line_id.map({l: i for i, l in enumerate(lines)})
    pat = pat.sort_values(["line_rank", "direction", "trips"], ascending=[True, True, False])
    cols = ["line_id", "direction", "pattern", "role", "trips", "share", "stops",
            "first_dep", "last_dep", "in_peak", "enter_service", "leave_service"]
    pat[cols].assign(first_dep=pat.first_dep.map(hhmm), last_dep=pat.last_dep.map(hhmm)).to_csv(
        out / "patterns.csv", index=False, float_format="%.3f")

    # Trips per hour that are full vs short, per line and direction.
    trips["hour"] = trips.start_min // 60
    trips = trips.join(pat.set_index(key).role, on=key)
    hourly = trips.groupby(["line_id", "direction", "hour", "role"]).size().unstack(fill_value=0)
    hourly.to_csv(out / "hourly.csv")

    # Timeline: one row per pattern, a tick at every trip start.
    has_short = [l for l in lines if (pat[(pat.line_id == l)].role == "short-turn service").any()]
    rows = pat[pat.line_id.isin(has_short) & (pat.role != "positioning")]
    colors = {"terminus to terminus": SERIES[0], "short-turn service": SERIES[1], "express": SERIES[2]}
    fig, ax = plt.subplots(figsize=(10, 0.24 * len(rows) + 1))
    labels = []
    for y, r in enumerate(rows.itertuples()):
        deps = trips[(trips.line_id == r.line_id) & (trips.direction == r.direction)
                     & (trips.pattern == r.pattern)].start_min
        ax.vlines(deps, y - 0.35, y + 0.35, color=colors[r.role], lw=0.8)
        labels.append(f"{r.line_id} {r.pattern}  ({r.trips})")
    ax.set_yticks(range(len(rows)), labels, fontsize=7)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    time_axis(ax, 4 * 60 + 30, 24 * 60, 60)
    ax.tick_params(axis="x", labelsize=7, rotation=90)
    for y in np.flatnonzero(rows.line_id.ne(rows.line_id.shift()).to_numpy())[1:]:
        ax.axhline(y - 0.5, color=INK_2, lw=0.5)
    handles = [plt.Line2D([], [], color=c, lw=2) for c in (SERIES[0], SERIES[1], SERIES[2])]
    ax.legend(handles, ["terminus to terminus", "short turn", "express (Line 6)"],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    ax.set_title("Trip starts by pattern (trips per day in brackets); positioning trips left out",
                 pad=24)
    save(fig, out / "short_turn_timeline.png")

    # Share of each line's trips that turn short, per rush-hour period.
    share = (trips[trips.line_id.isin(has_short)]
             .assign(short=lambda d: d.role == "short-turn service")
             .groupby(["line_id", "period"], observed=False).short.mean().unstack()
             .reindex(has_short)[PERIOD_ORDER])
    fig, ax = plt.subplots(figsize=(9, 3.4))
    width = 0.8 / len(PERIOD_ORDER)
    x = np.arange(len(has_short))
    for i, p in enumerate(PERIOD_ORDER):
        ax.bar(x + (i - 2) * width, share[p] * 100, width, label=p.replace("_", " "),
               color=["#9ec5f4", SERIES[0], "#86b6ef", "#184f95", "#cde2fb"][i],
               edgecolor="#fcfcfb", linewidth=0.8)
    ax.set_xticks(x, has_short)
    ax.set_ylabel("% of trips that are short turns")
    ax.set_title("How much of each line's service turns short, by time of day")
    ax.legend(ncol=5, loc="upper left")
    ax.set_ylim(0, 100)
    save(fig, out / "short_turn_share.png")

    # Where: share of the line's daytime trips that run over each segment,
    # with turn-back stations of short-turn services marked and Line 6
    # express skips drawn as open circles.
    smap = SubwayMap(map_json5, stops, lines_csv, closures_csv)
    day_ids = set(all_trips.trip_id[all_trips.start_min.between(*EDGE_OF_DAY)])
    s = stops[stops.trip_id.isin(day_ids)].sort_values(["trip_id", "seq"])
    nxt = s.groupby("trip_id").station.shift(-1)
    hops = pd.DataFrame({"line_id": s.line_id, "a": s.station, "b": nxt}).dropna()
    hops["seg"] = [tuple(sorted(p)) for p in zip(hops.a, hops.b)]
    per_seg = hops.groupby(["line_id", "seg"]).size()
    cover = per_seg / per_seg.groupby(level=0).transform("max")
    coverage = pd.DataFrame({"trips": per_seg, "share": cover}).reset_index()
    coverage[["a", "b"]] = pd.DataFrame(coverage.seg.tolist(), index=coverage.index)
    coverage.drop(columns="seg").to_csv(out / "coverage.csv", index=False, float_format="%.3f")
    cmap = mpl.colors.ListedColormap(["#9a3a12", "#d95926", "#eb6834", "#f2a07c", "#d6d5ce"])
    norm = mpl.colors.BoundaryNorm([0, 0.4, 0.6, 0.8, 0.95, 1.001], cmap.N)
    fig, ax = smap.figure(width=11)
    segment_values(smap, {(r.line_id, r.a, r.b): r.share for r in coverage.itertuples()},
                   ax, cmap, norm, lw=3)
    svc = pat[pat.role == "short-turn service"]
    turns = {}
    for r in svc.itertuples():
        o, d = r.pattern.split("→")
        for st in (o, d):
            if st not in smap.route[r.line_id][:1] + smap.route[r.line_id][-1:]:
                turns[(r.line_id, st)] = turns.get((r.line_id, st), 0) + r.trips
    placed = set()
    for (line, st), n in sorted(turns.items(), key=lambda kv: -kv[1]):
        x, y = smap.xy(line, st)
        ax.scatter([x], [y], s=26, color=INK, edgecolor=SURFACE, linewidth=0.8, zorder=5)
        if st not in placed:
            ax.annotate(st, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=6.5,
                        color=INK, zorder=6)
            placed.add(st)
    skipped = stops[stops.passed == 1].groupby(["line_id", "station"]).size()
    for (line, st), _ in skipped.items():
        x, y = smap.xy(line, st)
        ax.scatter([x], [y], s=34, facecolor=SURFACE, edgecolor=SERIES[2], linewidth=1.6, zorder=5)
    handles = [mpl.patches.Patch(color=cmap(i), label=lab) for i, lab in
               enumerate(["under 40%", "40–60%", "60–80%", "80–95%", "95% or more"])]
    handles += [plt.Line2D([], [], ls="", marker="o", color=INK, label="short-turn station"),
                plt.Line2D([], [], ls="", marker="o", mfc=SURFACE, mec=SERIES[2], mew=1.6,
                           label="passed by Line 6 expresses")]
    ax.legend(handles=handles, loc="lower left", fontsize=7, title="share of the line's trains, 07–22h",
              title_fontsize=7)
    ax.set_title("Where trains turn short: segments served by fewer of the line's trains", loc="left")
    save(fig, out / "coverage_map.png")

    # Positioning patterns, with how many of their trips start or end a working.
    posn = pat[pat.role == "positioning"].assign(
        first_dep=lambda d: d.first_dep.map(hhmm), last_dep=lambda d: d.last_dep.map(hhmm))
    posn = posn[["line_id", "direction", "pattern", "trips", "first_dep", "last_dep",
                 "enter_service", "leave_service"]]
    posn.to_csv(out / "positioning.csv", index=False)
    n_pos = int(posn.trips.sum())
    n_enter = int(posn.enter_service.sum())
    n_leave = int(posn.leave_service.sum())
    n_both = int(trips[trips.pattern.isin(posn.pattern) & trips.enters_service
                       & trips.leaves_service].shape[0])
    pat = pat.copy()
    md = [
        "## Short-turn trains\n",
        f"Patterns other than terminus to terminus that run at least {REGULAR} times a day in one "
        f"direction count as short-turn services; the rest are positioning trips. "
        f"{len(svc)} short-turn services on {svc.line_id.nunique()} lines: "
        f"{', '.join(has_short)}. The loop lines (L2, L10) are left out. "
        "The table lists short-turn and express patterns; patterns.csv has all of them.\n",
        "![](short_turn_timeline.png)\n",
        "![](short_turn_share.png)\n",
        "![](coverage_map.png)\n",
        md_table(pat.loc[pat.role.isin(["short-turn service", "express"]),
                         ["line_id", "direction", "pattern", "role", "trips", "share",
                          "first_dep", "last_dep", "in_peak"]]
                 .assign(first_dep=lambda d: d.first_dep.map(hhmm),
                         last_dep=lambda d: d.last_dep.map(hhmm)), "{:.2f}"),
        "",
        "### Positioning trips\n",
        f"{n_pos} trips in {len(posn)} patterns. Chaining trips into train workings, "
        f"{n_enter} of them are a train's first trip (it enters service there, from a depot or "
        f"siding) and {n_leave} its last (it then leaves service); {n_both} are both, a single "
        f"trip between two depots. The rest sit inside a working, for example a train put in at "
        "rush hour that turns back at an intermediate station.\n",
        md_table(posn, "{:.2f}"),
        "",
    ]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--explore", required=True)
    parser.add_argument("--lines", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--loop-closures", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    main(args.explore, args.lines, args.map, args.loop_closures, args.out)

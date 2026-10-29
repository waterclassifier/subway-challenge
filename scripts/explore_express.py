"""Skip-stop (express) trains: where they skip, how much they save, who waits.

In this snapshot only Line 6 runs trains that pass stations. The schedule
marks a passed station with "(", and a regular train that waits to be
overtaken gets two entries at one station (arrival "07:40-", departure
07:43), which explore_trips turns into one stop with hold_min = 3. Those are
the only departure times in the data; every other stop has one time, its
arrival.

For each express trip the "express section" runs from its last stop before
the first passed station to its first stop after the last one. Regular trips
are compared over the same section, leaving its first station within
WINDOW minutes of the express. A section time runs from leaving the first
station (its arrival time, or its departure if the train is held there) to
arriving at the last, so holds inside the section count against it.

Station distances (for the time-distance diagram) come from the line's
Beijing-Subway-Tools file; only its station list is read.

Outputs (in --out): express_trips.csv, overtakes.csv, served.csv,
marey_*.png, express_saving.png and summary.md.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

from explore_common import (
    INK_2, MUTED, SERIES, hhmm, load, md_table, plt, save, time_axis,
)

WINDOW = 30
LINE = "L6"


def station_km(tools_line_file):
    """Cumulative km along the line, from the stations list of a .json5 file."""
    text = Path(tools_line_file).read_text(encoding="utf-8")
    block = text[text.index("stations:"):text.index("train_routes:")]
    km, total = {}, 0.0
    for name, dist in re.findall(r'\{name:\s*"([^"]+)"(?:,\s*dist:\s*(\d+))?', block):
        total += int(dist or 0) / 1000
        km[name] = total
    return km


def section_time(stops, a, b):
    """Leaving a to arriving at b for one trip, or None if it misses either."""
    s = stops.set_index("station")
    if a not in s.index or b not in s.index or s.loc[a, "passed"] or s.loc[b, "passed"]:
        return None
    leave = s.loc[a, "time_min"] + s.loc[a, "hold_min"]
    return s.loc[b, "time_min"] - leave, leave


def main(explore_dir, tools_line_file, out_dir):
    out = Path(out_dir)
    trips, stops = load(explore_dir)
    trips = trips[trips.line_id == LINE]
    stops = stops[stops.line_id == LINE]
    by_trip = {k: g.sort_values("seq") for k, g in stops.groupby("trip_id")}
    km = station_km(tools_line_file)

    held = stops[stops.hold_min > 0]
    regular = trips[trips.express == 0]

    express_rows, compare_rows = [], []
    for e in trips[trips.express == 1].itertuples():
        st = by_trip[e.trip_id]
        passed = st[st.passed == 1]
        first_pass, last_pass = passed.seq.min(), passed.seq.max()
        a = st[st.seq == first_pass - 1].station.item()
        b = st[st.seq == last_pass + 1].station.item()
        e_time, e_dep = section_time(st, a, b)
        reg_times = []
        for r in regular[regular.direction == e.direction].itertuples():
            res = section_time(by_trip[r.trip_id], a, b)
            if res and abs(res[1] - e_dep) <= WINDOW:
                reg_times.append(res[0])
                compare_rows.append([e.trip_id, r.trip_id, res[0], res[1] - e_dep])
        reg_median = pd.Series(reg_times).median()
        express_rows.append([
            e.trip_id, e.direction, e.origin, e.destination, hhmm(e.start_min),
            a, b, len(passed), " ".join(passed.station), e_time, reg_median,
            reg_median - e_time, len(reg_times),
        ])
    ex = pd.DataFrame(express_rows, columns=[
        "trip_id", "direction", "origin", "destination", "departs", "section_from", "section_to",
        "n_passed", "passed", "express_min", "regular_median_min", "saved_min", "n_regular"])
    ex.to_csv(out / "express_trips.csv", index=False)

    # Overtakes: a regular trip held at a station while an express passes it,
    # or calls there too (cross-platform overtake).
    over_rows = []
    ex_stops = stops[stops.trip_id.isin(ex.trip_id)]
    for h in held.itertuples():
        by = ex_stops[(ex_stops.direction == h.direction) & (ex_stops.station == h.station)
                      & ex_stops.time_min.between(h.time_min, h.time_min + h.hold_min)]
        how = "/".join(sorted({"passes" if p else "stops" for p in by.passed}))
        over_rows.append([h.trip_id, h.direction, h.station, hhmm(h.time_min), h.hold_min,
                          " ".join(by.trip_id) or "-", how or "-"])
    ov = pd.DataFrame(over_rows, columns=["held_trip", "direction", "station", "arrives",
                                          "hold_min", "overtaken_by", "express_at_station"])
    ov.to_csv(out / "overtakes.csv", index=False)

    # Do held trains lose time overall? Compare full-length regular trips by
    # whether they were held, within the hours express trains run.
    full = regular[regular.kind == "full"].copy()
    full["held"] = full.trip_id.isin(held.trip_id)
    windows = ex.groupby("direction").departs.agg(["min", "max"])
    rows = []
    for d, (lo, hi) in windows.iterrows():
        lo_m = int(lo[:2]) * 60 + int(lo[3:]) - 60
        hi_m = int(hi[:2]) * 60 + int(hi[3:]) + 60
        f = full[(full.direction == d) & full.start_min.between(lo_m, hi_m)]
        for is_held, g in f.groupby("held"):
            rows.append([d, "held for an overtake" if is_held else "not held", len(g),
                         g.duration_min.mean(), g.duration_min.min(), g.duration_min.max()])
    held_cmp = pd.DataFrame(rows, columns=["direction", "regular trips", "n", "mean_min",
                                           "min_min", "max_min"])

    # Which stations each pattern serves.
    order = list(by_trip[trips.loc[trips.n_stops.idxmax(), "trip_id"]].station)
    served = pd.DataFrame(index=order)
    for (d, passed_set), g in ex.groupby(["direction", "passed"]):
        skipped = set(passed_set.split())
        span = by_trip[g.trip_id.iloc[0]].station.tolist()
        served[f"{d} {g.section_from.iloc[0]}–{g.section_to.iloc[0]} ({len(g)})"] = [
            "skip" if s in skipped else ("stop" if s in span else "") for s in order]
    served.index.name = "station"
    served.to_csv(out / "served.csv")

    # Time-distance diagrams: the first 100 minutes of each direction's main
    # express run (the pattern with the most trips).
    main_run = ex.groupby(["direction", "passed"]).departs.agg(["size", "min"]).reset_index()
    main_run = main_run.sort_values("size").groupby("direction").last()
    for d, first in main_run["min"].items():
        lo_m = int(first[:2]) * 60 + int(first[3:]) - 15
        hi_m = lo_m + 130
        fig, ax = plt.subplots(figsize=(11, 6))
        for t in trips[(trips.direction == d) & trips.start_min.between(lo_m - 100, hi_m)].itertuples():
            st = by_trip[t.trip_id]
            x = []
            y = []
            for s in st.itertuples():
                x += [s.time_min, s.time_min + s.hold_min]
                y += [km[s.station]] * 2
            if t.express:
                color, lw, z = SERIES[2], 1.6, 3
            elif t.trip_id in set(held.trip_id):
                color, lw, z = SERIES[1], 1.2, 2
            else:
                color, lw, z = "#9ec5f4", 0.9, 1
            ax.plot(x, y, color=color, lw=lw, zorder=z)
        ax.set_yticks([km[s] for s in order], order, fontsize=6)
        ax.set_ylim(min(km.values()), max(km.values()))
        if d == trips.direction.iloc[0] and by_trip[trips.trip_id.iloc[0]].station.iloc[0] == order[0]:
            ax.invert_yaxis()
        time_axis(ax, lo_m, hi_m, 10)
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.grid(axis="y", lw=0.3)
        handles = [plt.Line2D([], [], color=c, lw=2) for c in (SERIES[2], SERIES[1], "#9ec5f4")]
        ax.legend(handles, ["express", "regular, held for an overtake", "regular"],
                  loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
        ax.set_title(f"Line 6 {d}, from {hhmm(lo_m)}: time-distance diagram (flat steps are holds)",
                     pad=24)
        save(fig, out / f"marey_{d}.png")

    fig, ax = plt.subplots(figsize=(8, 3.2))
    for i, (d, g) in enumerate(ex.groupby("direction")):
        ax.scatter([int(t[:2]) * 60 + int(t[3:]) for t in g.departs], g.saved_min,
                   color=SERIES[i], s=28, label=d, edgecolor="#fcfcfb", linewidth=1, zorder=3)
    time_axis(ax, 6 * 60, 21 * 60, 60)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("minutes saved over the section")
    ax.set_title(f"Express vs median regular trip over the same section (±{WINDOW} min)")
    ax.legend()
    save(fig, out / "express_saving.png")

    groups = (ex.groupby(["direction", "section_from", "section_to", "n_passed"])
              .agg(trips=("trip_id", "size"), first=("departs", "min"), last=("departs", "max"),
                   express_min=("express_min", "mean"), regular_min=("regular_median_min", "mean"),
                   saved_min=("saved_min", "mean"))
              .reset_index())
    groups["saved_per_skip"] = groups.saved_min / groups.n_passed
    # How often: share of the direction's trips that are express, over the
    # span from a pattern's first to last express departure.
    def span_counts(g):
        lo, hi = (int(t[:2]) * 60 + int(t[3:]) for t in (g["first"], g["last"]))
        w = trips[(trips.direction == g.direction) & trips.start_min.between(lo, hi)]
        return pd.Series({"all_trips_in_span": len(w),
                          "express_share": g.trips / len(w),
                          "held_in_span": int(ov[(ov.direction == g.direction)].arrives.between(
                              g["first"], hhmm(hi + 90)).sum())})
    groups = groups.join(groups.apply(span_counts, axis=1))
    md = [
        "## Skip-stop (express) trains\n",
        f"Only {LINE} has trains that pass stations: {len(ex)} express trips. Express section = last "
        "stop before the first passed station to the first stop after the last one; regular trips "
        f"are compared over the same section, leaving its first station within ±{WINDOW} min.\n",
        md_table(groups, "{:.1f}"),
        "",
        "Regular full-length trips within an hour of the express window, held vs not held:\n",
        md_table(held_cmp, "{:.1f}"),
        "",
        f"Holds: {len(ov)} regular trips wait at "
        f"{', '.join(f'{s} ({n})' for s, n in ov.station.value_counts().items())}, "
        f"for {', '.join(f'{m} min ({n})' for m, n in ov.hold_min.value_counts().sort_index().items())}; "
        f"{(ov.overtaken_by != '-').sum()} of them are overtaken by an express during the hold "
        f"({', '.join(f'{k}: {v}' for k, v in ov.express_at_station.value_counts().items())}).\n",
        "Stations served (stop) and passed (skip) per express pattern, with trips per day:\n",
        md_table(served.reset_index()),
        "",
        "![](express_saving.png)\n",
        *[f"![](marey_{d}.png)\n" for d in main_run.index],
    ]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--explore", required=True)
    parser.add_argument("--tools-line", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    main(args.explore, args.tools_line, args.out)

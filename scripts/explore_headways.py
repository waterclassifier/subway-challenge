"""How often trains come: headways by line, station, direction and time of day.

The headway at a station is the gap between consecutive trains stopping there
in one direction (a trip's last stop doesn't count; you can't board there).
Three views of it:
- headway strips: per line and direction, a station-by-time grid of the
  current headway, i.e. the gap between the last train before that moment and
  the next one after it. Short turns show up as bands of stations with longer
  gaps, rush hours as columns of short ones, and the service day as its extent.
- frequency maps: the subway map at four times of day, each segment coloured
  by trains per hour per direction.
- waits: the expected wait for someone arriving at a random moment,
  sum(h^2) / (2 * sum(h)), which is longer than half the mean headway when
  gaps are uneven, and the longest gap between 07:00 and 22:00.

Outputs (in --out): stations.csv (per station and direction), lines.csv,
segments.csv (trains per hour per segment and window), strips/<line>.png,
frequency_maps.png, wait_by_hour.png, max_gap.png and summary.md.
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

from explore_common import (
    BLUE_RAMP, INK, INK_2, SERIES, hhmm, line_order, load, md_table, period_of, plt, save, time_axis,
)
from explore_map import SubwayMap, segment_values

DAY = (7 * 60, 22 * 60)
STRIP_STEP = 5  # minutes per column of a headway strip
# Short gaps dark, long gaps light: a strip reads like a density of trains.
GAP_BOUNDS = [0, 2, 3, 4, 6, 8, 12, 30]
GAP_COLORS = BLUE_RAMP[::-1]
MAP_WINDOWS = [("06:00–07:00", 6 * 60), ("08:00–09:00", 8 * 60),
               ("12:00–13:00", 12 * 60), ("22:00–23:00", 22 * 60)]


def expected_wait(h):
    h = np.asarray(h, dtype=float)
    return (h ** 2).sum() / (2 * h.sum()) if len(h) else np.nan


def current_headway(times, grid):
    """Gap between the last train before and the first train after each grid time."""
    times = np.sort(np.asarray(times))
    i = np.searchsorted(times, grid, side="right")
    out = np.full(len(grid), np.nan)
    ok = (i > 0) & (i < len(times))
    out[ok] = times[i[ok]] - times[i[ok] - 1]
    return out


def strips(calls, smap, line, path):
    """Station-by-time headway grid, one panel per direction."""
    cmap = mpl.colors.ListedColormap(GAP_COLORS)
    norm = mpl.colors.BoundaryNorm(GAP_BOUNDS, cmap.N)
    grid = np.arange(4 * 60 + 30, 24 * 60 + 30, STRIP_STEP)
    lc = calls[calls.line_id == line]
    dirs = sorted(lc.direction.unique())
    route = smap.route[line]
    fig, axes = plt.subplots(1, len(dirs), figsize=(11, 0.17 * len(route) + 1.4), squeeze=False)
    for ax, d in zip(axes[0], dirs):
        dc = lc[lc.direction == d]
        # Stations in this direction's running order.
        first_trip = dc.groupby("trip_id").size().idxmax()
        seq = dc[dc.trip_id == first_trip].sort_values("seq").station.tolist()
        order = seq + [s for s in route if s not in seq]
        m = np.array([current_headway(dc[dc.station == s].time_min, grid) for s in order])
        ax.imshow(m, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest",
                  extent=(grid[0], grid[-1] + STRIP_STEP, len(order) - 0.5, -0.5))
        ax.set_yticks(range(len(order)), order, fontsize=6)
        ax.grid(False)
        time_axis(ax, 5 * 60, 24 * 60, 120)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_title(f"{line} {d}", fontsize=9)
    fig.colorbar(mpl.cm.ScalarMappable(norm, cmap), ax=axes[0].tolist(), shrink=0.8, pad=0.01,
                 label="minutes to the next train")
    save(fig, path)


def main(explore_dir, lines_csv, map_json5, closures_csv, out_dir):
    out = Path(out_dir)
    trips, stops = load(explore_dir)
    lines = [l for l in line_order(lines_csv) if l in set(trips.line_id)]
    smap = SubwayMap(map_json5, stops, lines_csv, closures_csv)
    calls = stops[stops.passed == 0].sort_values(["line_id", "direction", "station", "time_min"])
    last_seq = calls.groupby("trip_id").seq.transform("max")
    calls = calls[calls.seq < last_seq].copy()
    calls["gap"] = calls.groupby(["line_id", "direction", "station"]).time_min.diff()
    calls["hour"] = calls.time_min // 60
    gaps = calls.dropna(subset=["gap"])
    day = gaps[gaps.time_min.between(*DAY)]

    key = ["line_id", "direction", "station"]
    st = calls.groupby(key).agg(first=("time_min", "min"), last=("time_min", "max"),
                                trains=("trip_id", "size")).reset_index()
    st = st.join(day.groupby(key).gap.max().rename("max_gap_day"), on=key)
    for p in ["am_peak", "midday", "pm_peak"]:
        sel = gaps[period_of(gaps.time_min) == p]
        st = st.join(sel.groupby(key).gap.apply(expected_wait).rename(f"wait_{p}"), on=key)
    line_trains = st.groupby(["line_id", "direction"]).trains.transform("max")
    st["share_of_trains"] = st.trains / line_trains
    st.assign(first=st["first"].map(hhmm), last=st["last"].map(hhmm)).to_csv(
        out / "stations.csv", index=False, float_format="%.2f")

    (out / "strips").mkdir(exist_ok=True)
    for line in lines:
        strips(calls, smap, line, out / "strips" / f"{line}.png")

    # Trains per hour per segment and direction, averaged over the two directions.
    s = stops.sort_values(["trip_id", "seq"])
    nxt = s.groupby("trip_id").station.shift(-1)
    hops = pd.DataFrame({"line_id": s.line_id, "a": s.station, "b": nxt, "t": s.time_min}).dropna()
    hops["seg"] = [tuple(sorted(p)) for p in zip(hops.a, hops.b)]
    rows = []
    for label, start in MAP_WINDOWS:
        w = hops[hops.t.between(start, start + 59)]
        for (line, seg), n in w.groupby(["line_id", "seg"]).size().items():
            rows.append([label, line, seg[0], seg[1], n / 2])
    seg = pd.DataFrame(rows, columns=["window", "line_id", "a", "b", "trains_per_hour"])
    seg.to_csv(out / "segments.csv", index=False)
    # Discrete steps so a map can be read against the legend; light = rare.
    cmap = mpl.colors.ListedColormap(["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
                                      "#184f95", "#0d366b"])
    norm = mpl.colors.BoundaryNorm([0, 6, 8, 10, 12, 15, 20, 32], cmap.N)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), layout="constrained")
    for ax, (label, _) in zip(axes.flat, MAP_WINDOWS):
        sw = seg[seg.window == label]
        smap.frame(ax)
        segment_values(smap, {(r.line_id, r.a, r.b): r.trains_per_hour for r in sw.itertuples()},
                       ax, cmap, norm, lw=1.8)
        ax.set_title(label, loc="left")
    fig.colorbar(mpl.cm.ScalarMappable(norm, cmap), ax=axes.ravel().tolist(), shrink=0.5,
                 label="trains per hour, per direction")
    fig.suptitle("How often trains run on each stretch of track", x=0.02, ha="left",
                 fontweight="bold")
    save(fig, out / "frequency_maps.png")

    # Per line and hour: expected wait over all station-directions of the line.
    lh = gaps.groupby(["line_id", "hour"]).gap.apply(expected_wait).unstack().reindex(lines)
    hours = list(range(5, 24))
    lh = lh.reindex(columns=hours)
    fig, ax = plt.subplots(figsize=(10, 0.28 * len(lines) + 1.2))
    bounds = [0, 1.5, 2, 3, 4, 6, 10, 60]
    wcmap = mpl.colors.ListedColormap(BLUE_RAMP)
    im = ax.imshow(lh.to_numpy(), aspect="auto", cmap=wcmap,
                   norm=mpl.colors.BoundaryNorm(bounds, wcmap.N))
    ax.set_xticks(range(len(hours)), [f"{h:02d}" for h in hours])
    ax.set_yticks(range(len(lines)), lines)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(lh.to_numpy()):
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center", va="center",
                    fontsize=6, color="#ffffff" if v >= 4 else INK)
    ax.set_xlabel("hour")
    ax.set_title("Expected wait for a train (minutes), random arrival, averaged over stations")
    fig.colorbar(im, ax=ax, shrink=0.6, label="minutes")
    save(fig, out / "wait_by_hour.png")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(st.max_gap_day.dropna(), bins=np.arange(0, st.max_gap_day.max() + 2, 1), color=SERIES[0])
    ax.set_xlabel(f"longest gap between trains, {hhmm(DAY[0])}–{hhmm(DAY[1])} (minutes)")
    ax.set_ylabel("station-directions")
    ax.set_title("Worst-case wait at each station and direction")
    save(fig, out / "max_gap.png")

    lt = st.groupby("line_id").agg(
        wait_am_peak=("wait_am_peak", "median"), wait_midday=("wait_midday", "median"),
        wait_pm_peak=("wait_pm_peak", "median"), max_gap_day=("max_gap_day", "max"),
        min_share_of_trains=("share_of_trains", "min")).reindex(lines).reset_index()
    lt.to_csv(out / "lines.csv", index=False, float_format="%.2f")
    worst = st.sort_values(["max_gap_day", "line_id"], ascending=[False, True]).drop_duplicates(
        "line_id").head(10)
    sparse = st.sort_values(["share_of_trains", "line_id", "station"]).head(12)
    md = [
        "## How often trains come\n",
        f"- Median expected wait over all station-directions: "
        f"AM peak {st.wait_am_peak.median():.1f} min, midday {st.wait_midday.median():.1f} min, "
        f"PM peak {st.wait_pm_peak.median():.1f} min.",
        f"- Longest daytime gap ({hhmm(DAY[0])}–{hhmm(DAY[1])}): {st.max_gap_day.max():.0f} min; "
        f"median over station-directions {st.max_gap_day.median():.0f} min.",
        "- Headway strips for every line are in `strips/`; three examples below.",
        "",
        "![](frequency_maps.png)\n",
        "![](strips/L6.png)\n",
        "![](strips/CP.png)\n",
        "![](strips/L4.png)\n",
        "![](wait_by_hour.png)\n",
        "![](max_gap.png)\n",
        md_table(lt, "{:.1f}"),
        "",
        "Longest daytime gap per line, worst lines first (one station shown per line):\n",
        md_table(worst[key + ["trains", "max_gap_day", "wait_midday"]], "{:.1f}"),
        "",
        "Stations served by the smallest share of their line's trains (short turns and expresses "
        "pass them by):\n",
        md_table(sparse[key + ["trains", "share_of_trains", "wait_midday", "wait_am_peak"]], "{:.2f}"),
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

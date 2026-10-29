"""How many trains run during a weekday, and when.

A train is en route from its arrival at the first stop of a trip (where it
starts boarding) to its arrival at the last. Each stop has one time, read as
the arrival, so that is the span from a trip's first time to its last. Time
spent standing at a terminus between trips is not counted. On the loop lines
each lap is its own trip, so a train is briefly uncounted between laps.

Outputs (in --out): en_route.csv (trains en route per line and minute),
lines.csv (per-line counts), en_route_total.png, en_route_lines.png,
peak_ratio_map.png, trips_per_line.png and summary.md.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from explore_common import INK, SERIES, hhmm, line_order, load, md_table, plt, save, time_axis
from explore_map import SubwayMap, line_values

START, END = 4 * 60, 26 * 60
MIDDAY = (11 * 60, 14 * 60)


def en_route(trips, lines):
    minutes = np.arange(START, END)
    counts = {}
    for line in lines:
        t = trips[trips.line_id == line]
        # +1 when a trip reaches its first stop, -1 when it reaches its last, then a running sum.
        delta = np.zeros(END - START + 1, dtype=int)
        np.add.at(delta, t.start_min - START, 1)
        np.add.at(delta, t.end_min - START, -1)
        counts[line] = np.cumsum(delta)[:-1]
    return pd.DataFrame(counts, index=pd.Index(minutes, name="minute"))


def main(explore_dir, lines_csv, map_json5, closures_csv, out_dir):
    out = Path(out_dir)
    trips, stops = load(explore_dir)
    lines = [l for l in line_order(lines_csv) if l in set(trips.line_id)]
    occ = en_route(trips, lines)
    occ.to_csv(out / "en_route.csv")
    total = occ.sum(axis=1)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.fill_between(occ.index, total, color=SERIES[0], lw=0)
    peak_t = int(total.idxmax())
    ax.annotate(f"{total.max()} trains at {hhmm(peak_t)}", (peak_t, total.max()),
                xytext=(10, 4), textcoords="offset points", color=INK, fontsize=8)
    time_axis(ax, START, END)
    ax.set_ylabel("trains en route")
    ax.set_title("Trains en route at each minute of a weekday, whole network")
    save(fig, out / "en_route_total.png")

    # Small multiples per line, own y-scale, so the shape of each line's day shows.
    n = len(lines)
    cols = 4
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(11, 1.5 * rows), sharex=True)
    # A 5-minute mean hides the one-minute flicker of trips handing over at termini.
    smooth = occ.rolling(5, center=True, min_periods=1).mean()
    for ax, line in zip(axes.flat, lines):
        ax.fill_between(smooth.index, smooth[line], color=SERIES[0], lw=0)
        ax.set_title(f"{line} · peak {occ[line].max()}", fontsize=8, fontweight="normal")
        ax.set_ylim(0, occ[line].max() * 1.15)
        ax.tick_params(labelsize=7)
        time_axis(ax, 5 * 60, 25 * 60, 240)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle("Trains en route per line, 5-minute mean (own y-scale per panel)", x=0.01, ha="left",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()
    save(fig, out / "en_route_lines.png")

    # Trips per line, by kind.
    kinds = ["full", "short", "loop"]
    per_line = trips.groupby(["line_id", "kind"]).size().unstack(fill_value=0).reindex(lines)
    per_line = per_line.reindex(columns=kinds, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    bottom = np.zeros(len(lines))
    for k, color in zip(kinds, [SERIES[0], SERIES[1], SERIES[2]]):
        ax.bar(lines, per_line[k], bottom=bottom, color=color, label=k, width=0.75,
               edgecolor="#fcfcfb", linewidth=1)
        bottom += per_line[k].to_numpy()
    ax.set_ylabel("trips per weekday (both directions)")
    ax.set_title("Weekday trips per line, by kind")
    ax.tick_params(axis="x", rotation=90)
    ax.legend()
    save(fig, out / "trips_per_line.png")

    mid = occ.loc[MIDDAY[0]:MIDDAY[1]]
    table = pd.DataFrame({
        "line": lines,
        "trips": [int((trips.line_id == l).sum()) for l in lines],
        "short_trips": [int(((trips.line_id == l) & (trips.kind == "short")).sum()) for l in lines],
        "peak_en_route": [int(occ[l].max()) for l in lines],
        "peak_at": [hhmm(occ[l].idxmax()) for l in lines],
        "midday_en_route": [int(round(mid[l].mean())) for l in lines],
    })
    table["peak_over_midday"] = (table.peak_en_route / table.midday_en_route).round(2)
    table.to_csv(out / "lines.csv", index=False)

    smap = SubwayMap(map_json5, stops, lines_csv, closures_csv)
    line_values(smap, dict(zip(table.line, table.peak_over_midday)), out / "peak_ratio_map.png",
                title="Trains en route at the line's peak, per train en route midday",
                label="peak / midday", fmt="{:.1f}")

    am = total.loc[6 * 60:10 * 60]
    pm = total.loc[16 * 60:20 * 60]
    first = trips.start_min.min()
    last = trips.end_min.max()
    md = [
        "## Trains en route\n",
        f"- Weekday trips: **{len(trips)}** ({', '.join(f'{k} {v}' for k, v in trips.kind.value_counts().items())}). "
        "The loop lines are cut into one trip per lap, so the trip count overstates trains there.",
        f"- First train at its first stop {hhmm(first)}, last arrival {hhmm(last)}.",
        f"- En route at once: morning peak **{am.max()}** at {hhmm(am.idxmax())}, "
        f"evening peak **{pm.max()}** at {hhmm(pm.idxmax())}, "
        f"midday ({hhmm(MIDDAY[0])}–{hhmm(MIDDAY[1])}) about **{int(round(mid.sum(axis=1).mean()))}**.",
        "",
        "![](en_route_total.png)\n",
        "![](en_route_lines.png)\n",
        "![](peak_ratio_map.png)\n",
        "![](trips_per_line.png)\n",
        md_table(table, "{:.2f}"),
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

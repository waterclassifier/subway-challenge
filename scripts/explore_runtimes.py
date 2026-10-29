"""How consistent is the time from a station to the next, by direction and time of day.

A hop is one trip going from a stop to its next stop. The schedule has one
time per stop, its arrival, so a hop's time is arrival to arrival: the dwell
at the first stop plus the run to the next. Left out are hops next to a
passed station (an express's run past it has no time of its own) and hops
leaving a stop where a Line 6 train is held for an overtake (the hold is not
running time).

Schedule times are whole minutes, so a hop that always takes 2.5 minutes
shows up as a mix of 2s and 3s: a spread of one minute is what rounding alone
produces. "Consistent" below means a spread of at most one minute.

Outputs (in --out): hops.csv (per directed hop), pairs.csv (both directions
of a station pair side by side), full_trips.csv, hop_spread.png,
direction_delta.png, trip_duration_by_time.png, period_slowdown.png and
summary.md.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from explore_common import (
    BLUE_RAMP, INK, INK_2, MUTED, PERIOD_ORDER, SERIES, line_order, load, md_table, period_of,
    plt, save, time_axis,
)


def hops_of(stops):
    s = stops.sort_values(["line_id", "direction", "trip_id", "seq"])
    nxt = s.groupby("trip_id").shift(-1)
    h = pd.DataFrame({
        "line_id": s.line_id, "direction": s.direction, "trip_id": s.trip_id,
        "from_st": s.station, "to_st": nxt.station,
        "time_min": s.time_min, "minutes": nxt.time_min - s.time_min,
        "skip": (s.passed == 1) | (nxt.passed == 1) | (s.hold_min > 0),
    })
    h = h[h.to_st.notna() & ~h.skip].drop(columns="skip")
    h["minutes"] = h.minutes.astype(int)
    h["period"] = period_of(h.time_min)
    return h


def main(explore_dir, lines_csv, out_dir):
    out = Path(out_dir)
    trips, stops = load(explore_dir)
    lines = [l for l in line_order(lines_csv) if l in set(trips.line_id)]
    hops = hops_of(stops)
    key = ["line_id", "from_st", "to_st"]

    g = hops.groupby(key).minutes
    hop = g.agg(n="size", mean="mean", std="std", min="min", max="max").reset_index()
    hop["spread"] = hop["max"] - hop["min"]
    mode = hops.groupby(key).minutes.agg(lambda m: m.value_counts(normalize=True).iloc[0])
    hop = hop.join(mode.rename("share_at_mode"), on=key)
    by_period = hops.groupby(key + ["period"], observed=True).minutes.mean().unstack()
    hop = hop.join(by_period.add_prefix("mean_"), on=key)
    # Real variation (beyond rounding) is easier to see in the share of trips
    # more than a minute off the hop's median.
    med = hops.groupby(key).minutes.transform("median")
    hops["off"] = (hops.minutes - med).abs() > 1
    hop = hop.join(hops.groupby(key).off.mean().rename("share_off_by_2plus"), on=key)
    hop = hop[hop.n >= 20]  # drop hops only a few positioning trips make
    hop.sort_values(key).to_csv(out / "hops.csv", index=False, float_format="%.3f")

    # Pair each hop with its reverse.
    rev = hop.rename(columns={"from_st": "to_st", "to_st": "from_st"})
    pairs = hop.merge(rev[key + ["mean", "n"]], on=key, suffixes=("_ab", "_ba"))
    pairs = pairs[pairs.from_st < pairs.to_st]
    pairs["delta"] = pairs.mean_ab - pairs.mean_ba
    pairs[key + ["n_ab", "n_ba", "mean_ab", "mean_ba", "delta"]].to_csv(
        out / "pairs.csv", index=False, float_format="%.3f")

    # Whole-trip times of terminus-to-terminus trips, which average out the
    # rounding of individual hops.
    full = trips[trips.kind == "full"].copy()
    full["period"] = period_of(full.start_min)
    full.to_csv(out / "full_trips.csv", index=False)

    # Figure 1: how spread out is a hop's time?
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    counts = hop.spread.clip(upper=5).value_counts().sort_index()
    axes[0].bar([str(i) if i < 5 else "5+" for i in counts.index], counts.values / len(hop) * 100,
                color=[SERIES[0] if i <= 1 else SERIES[1] for i in counts.index], width=0.7)
    axes[0].set_xlabel("max minus min over all trips (minutes)")
    axes[0].set_ylabel("% of directed hops")
    axes[0].set_title("Spread of each hop's time (≤ 1 is rounding)")
    axes[1].hist(hop.share_off_by_2plus * 100, bins=np.arange(0, 51, 2), color=SERIES[0])
    axes[1].set_xlabel("% of a hop's trips more than 1 min off its median")
    axes[1].set_ylabel("directed hops")
    axes[1].set_title("How often a hop is really off")
    save(fig, out / "hop_spread.png")

    # Figure 2: A→B vs B→A.
    trip_dir = (full.groupby(["line_id", "direction"]).duration_min.median().unstack())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    axes[0].hist(pairs.delta, bins=np.arange(-3, 3.01, 0.25), color=SERIES[0])
    axes[0].set_xlabel("mean(A→B) minus mean(B→A), minutes")
    axes[0].set_ylabel("station pairs")
    axes[0].set_title("Direction difference per station pair")
    line_delta = []
    for l in lines:
        if l in trip_dir.index:
            v = trip_dir.loc[l].dropna()
            if len(v) == 2:
                line_delta.append((l, v.iloc[0] - v.iloc[1], v.index[0], v.index[1]))
    ld = pd.DataFrame(line_delta, columns=["line", "delta", "dir_a", "dir_b"])
    axes[1].bar(ld.line, ld.delta, color=[SERIES[0] if d >= 0 else SERIES[1] for d in ld.delta])
    axes[1].axhline(0, color=MUTED, lw=0.8)
    axes[1].tick_params(axis="x", rotation=90)
    axes[1].set_ylabel("minutes")
    axes[1].set_title("Whole line: median full trip, one direction minus the other")
    save(fig, out / "direction_delta.png")

    # Figure 3: full-trip duration through the day, per line and direction.
    with_full = [l for l in lines if (full.line_id == l).any() and trips[trips.line_id == l].kind.ne("loop").all()]
    cols = 4
    rows = -(-len(with_full) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(11, 1.7 * rows), sharex=True)
    for ax, l in zip(axes.flat, with_full):
        f = full[full.line_id == l]
        for i, (d, fd) in enumerate(f.groupby("direction")):
            ax.plot(fd.start_min, fd.duration_min, ".", ms=2.5, color=SERIES[i], label=d)
        lo, hi = f.duration_min.min(), f.duration_min.max()
        ax.set_ylim(lo - 1, max(hi + 1, lo * 1.12))
        ax.set_title(f"{l} · {lo}–{hi} min", fontsize=8, fontweight="normal")
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper right", handletextpad=0.1, markerscale=2)
        time_axis(ax, 5 * 60, 24 * 60, 240)
    for ax in axes.flat[len(with_full):]:
        ax.axis("off")
    fig.suptitle("Terminus-to-terminus trip time (first to last arrival) by start time (each y-axis spans at least 12% of the fastest trip)",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    save(fig, out / "trip_duration_by_time.png")

    # Figure 4: % slower than the line's fastest period, from full trips.
    per = full.groupby(["line_id", "period"], observed=True).duration_min.mean().unstack()
    per = per.reindex([l for l in with_full])[PERIOD_ORDER]
    slow = (per.div(per.min(axis=1), axis=0) - 1) * 100
    fig, ax = plt.subplots(figsize=(6, 0.28 * len(slow) + 1))
    cmap = matplotlib_cmap()
    im = ax.imshow(slow.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=max(8, np.nanmax(slow.to_numpy())))
    ax.set_xticks(range(len(PERIOD_ORDER)), [p.replace("_", " ") for p in PERIOD_ORDER])
    ax.set_yticks(range(len(slow)), slow.index)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(slow.to_numpy()):
        if not np.isnan(v):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                    color="#ffffff" if v > 0.6 * im.get_clim()[1] else INK)
    ax.set_title("Full trips: % slower than the line's fastest period")
    save(fig, out / "period_slowdown.png")

    consistent = (hop.spread <= 1).mean()
    worst = hop.sort_values("share_off_by_2plus", ascending=False).head(15)
    big = (pairs.assign(abs_delta=pairs.delta.abs())
           .sort_values(["abs_delta"] + key, ascending=[False, True, True, True]).head(12))
    off_by_period = hops.groupby("period", observed=True).off.mean() * 100
    md = [
        "## Run time between adjacent stations\n",
        f"- {len(hop)} directed hops (≥ 20 trips each), {len(hops)} trip-hops.",
        f"- **{consistent:.0%}** of hops have a spread of at most one minute over the whole day, "
        "i.e. no variation beyond what whole-minute rounding produces. "
        f"Overall {hops.off.mean():.1%} of trip-hops are more than a minute off their hop's median "
        f"({', '.join(f'{p} {v:.1f}%' for p, v in off_by_period.items())}).",
        f"- A→B vs B→A: the mean difference per station pair is within ±0.5 min for "
        f"{(pairs.delta.abs() <= 0.5).mean():.0%} of pairs and within ±1 min for "
        f"{(pairs.delta.abs() <= 1).mean():.0%}. Times are arrival to arrival, so a hop includes the "
        "dwell at its first stop, and A→B and B→A include dwells at different stations; over a "
        "whole line that evens out.",
        "",
        "![](hop_spread.png)\n",
        "![](direction_delta.png)\n",
        "![](trip_duration_by_time.png)\n",
        "![](period_slowdown.png)\n",
        "Hops most often more than a minute off their median:\n",
        md_table(worst[key + ["n", "mean", "min", "max", "share_off_by_2plus"]
                       + [f"mean_{p}" for p in PERIOD_ORDER if f"mean_{p}" in worst]], "{:.2f}"),
        "",
        "Station pairs with the largest direction difference:\n",
        md_table(big[key + ["mean_ab", "mean_ba", "delta"]], "{:.2f}"),
        "",
        "Whole-line direction difference (median full trip):\n",
        md_table(ld, "{:.1f}"),
        "",
    ]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")


def matplotlib_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("blue", ["#f0efec"] + BLUE_RAMP[1:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--explore", required=True)
    parser.add_argument("--lines", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    main(args.explore, args.lines, args.out)

"""Draw values on the subway map.

Station positions come from the official-map coordinates in
Beijing-Subway-Tools (maps/official.json5), so the drawing follows the
schematic map's layout; tracks are straight segments between adjacent
stations, which is close to the map but not its exact curves. Which
stations are adjacent comes from the trips themselves (stops.csv), plus the
loop-closing segments the source never runs.

Typical use from an explore_* script:

    smap = SubwayMap(args.map, stops, args.lines, args.loop_closures)
    line_values(smap, {"L1": 2.8, "L2": 2.5, ...}, out / "peak_ratio_map.png",
                title="Rush-hour trains per midday train", label="ratio")

For other drawings, start from smap.figure() and use smap.base(),
smap.draw_segments() and smap.mark_stations() on the axes it returns.
"""

import re
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

from explore_common import BLUE_RAMP, INK, INK_2, MUTED, SURFACE, plt, save

BASE_COLOR = "#d6d5ce"
COORD = re.compile(r'"([^"]+)":\s*\{(?:path_coords:\s*\{)?\s*x:\s*(\d+),\s*y:\s*(\d+)')


class SubwayMap:
    def __init__(self, map_json5, stops, lines_csv, loop_closures_csv=None):
        text = Path(map_json5).read_text(encoding="utf-8")
        self.coords = {name: (int(x), -int(y)) for name, x, y in COORD.findall(text)}
        meta = pd.read_csv(lines_csv)
        self.line_name = dict(zip(meta.line_id, meta.line_name_zh))
        self.lines = [l for l in meta.line_id if l in set(stops.line_id)]

        # Adjacent stations: consecutive entries of any trip, passed ones included.
        s = stops.sort_values(["trip_id", "seq"])
        nxt = s.groupby("trip_id").station.shift(-1)
        pairs = pd.DataFrame({"line_id": s.line_id, "a": s.station, "b": nxt}).dropna()
        self.segments = defaultdict(set)
        for line, a, b in pairs.drop_duplicates().itertuples(index=False):
            self.segments[line].add(tuple(sorted((a, b))))
        if loop_closures_csv:
            for c in pd.read_csv(loop_closures_csv).itertuples():
                self.segments[c.line_id].add(tuple(sorted((c.u_zh, c.v_zh))))
        longest = stops.groupby("trip_id").size().rename("n").reset_index().merge(
            stops[["trip_id", "line_id"]].drop_duplicates(), on="trip_id")
        longest = longest.sort_values(["n", "trip_id"]).groupby("line_id").last().trip_id
        self.route = {line: s[s.trip_id == trip].station.tolist() for line, trip in longest.items()}
        # Stations the map doesn't have (it is newer than some schedules) go
        # halfway between their neighbours on the line.
        self.placed = {}
        self.missing = sorted({(l, st) for l, segs in self.segments.items() for seg in segs
                               for st in seg if self.xy(l, st) is None})
        for line, station in self.missing:
            nbrs = [b if a == station else a for a, b in self.segments[line] if station in (a, b)]
            pts = [self.xy(line, n) for n in nbrs if self.xy(line, n) is not None]
            if not pts:
                raise SystemExit(f"{station} ({line}) and its neighbours are missing from {map_json5}")
            self.placed[(line, station)] = tuple(np.mean(pts, axis=0))

    def xy(self, line, station):
        """Map position of a station, using the line-specific entry where the map has one."""
        if (line, station) in getattr(self, "placed", {}):
            return self.placed[(line, station)]
        name = self.line_name[line]
        for key, pos in self.coords.items():
            if key.startswith(station + "(") and name.startswith(key[len(station) + 1:-1]):
                return pos
        return self.coords.get(station)

    def label_positions(self, min_gap=420):
        """One label spot per line: the stop nearest the middle of its longest
        trip that keeps clear of labels already placed."""
        spots = {}
        for line in self.lines:
            route = self.route[line]
            mid = len(route) // 2
            order = sorted(range(len(route)), key=lambda i: (abs(i - mid), i))
            pts = [self.xy(line, route[i]) for i in order]
            spots[line] = next(
                (p for p in pts if all(np.hypot(p[0] - q[0], p[1] - q[1]) >= min_gap
                                       for q in spots.values())), pts[0])
        return spots

    def frame(self, ax, pad=120):
        """Fit the axes to the map, equal scale, no axis lines."""
        xs = [x for x, _ in self.coords.values()]
        ys = [y for _, y in self.coords.values()]
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
        return (max(ys) - min(ys)) / (max(xs) - min(xs))

    def figure(self, width=10):
        fig, ax = plt.subplots()
        aspect = self.frame(ax)
        fig.set_size_inches(width, width * aspect)
        return fig, ax

    def draw_segments(self, ax, line, segments=None, **kw):
        segs = self.segments[line] if segments is None else segments
        lines = [[self.xy(line, a), self.xy(line, b)] for a, b in segs]
        style = {"linewidths": 2.2, "capstyle": "round", "joinstyle": "round"} | kw
        ax.add_collection(mpl.collections.LineCollection(lines, **style))

    def base(self, ax, color=BASE_COLOR, lw=1.6):
        for line in self.lines:
            self.draw_segments(ax, line, colors=color, linewidths=lw, zorder=1)

    def mark_stations(self, ax, line, stations, **kw):
        pts = np.array([self.xy(line, s) for s in stations])
        style = {"s": 18, "color": INK, "zorder": 4, "edgecolor": SURFACE, "linewidth": 0.8} | kw
        ax.scatter(pts[:, 0], pts[:, 1], **style)

    def label_line(self, ax, line, text=None, at=None, **kw):
        x, y = at or self.label_positions()[line]
        style = {"fontsize": 7, "color": INK, "ha": "center", "va": "center", "zorder": 6,
                 "bbox": {"boxstyle": "round,pad=0.2", "fc": SURFACE, "ec": MUTED, "lw": 0.5}} | kw
        ax.text(x, y, text or line, **style)


def line_values(smap, values, path, title, label, cmap=None, vmin=None, vmax=None, fmt="{:.1f}"):
    """Colour each line by its value; lines without a value stay grey."""
    # The ramp starts at a mid blue so even the lowest value stands out from the grey base.
    cmap = cmap or mpl.colors.LinearSegmentedColormap.from_list("ramp", ["#86b6ef"] + BLUE_RAMP[3:])
    vals = [v for v in values.values() if pd.notna(v)]
    norm = mpl.colors.Normalize(vmin if vmin is not None else min(vals),
                                vmax if vmax is not None else max(vals))
    fig, ax = smap.figure()
    smap.base(ax)
    # Draw low values first so the highest sit on top where lines share a corridor.
    for line in sorted(values, key=lambda l: values[l]):
        if pd.notna(values[line]):
            smap.draw_segments(ax, line, colors=[cmap(norm(values[line]))], linewidths=3.2, zorder=2)
    spots = smap.label_positions()
    for line in smap.lines:
        v = values.get(line)
        smap.label_line(ax, line, f"{line} {fmt.format(v)}" if pd.notna(v) else line,
                        at=spots[line], color=INK if pd.notna(v) else INK_2)
    fig.colorbar(mpl.cm.ScalarMappable(norm, cmap), ax=ax, shrink=0.5, pad=0.01, label=label)
    ax.set_title(title, loc="left")
    save(fig, path)


def segment_values(smap, values, ax, cmap, norm, lw=2.6):
    """Colour each segment, keyed (line_id, station, station) in any order."""
    smap.base(ax)
    by_line = {}
    for (line, a, b), v in values.items():
        if pd.notna(v):
            by_line.setdefault(line, []).append(((a, b), v))
    for line, items in by_line.items():
        segs = [s for s, _ in items]
        colors = [cmap(norm(v)) for _, v in items]
        smap.draw_segments(ax, line, segments=segs, colors=colors, linewidths=lw, zorder=2)

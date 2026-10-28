"""Build the model A station graph from a pinned train schedule snapshot.

Model A: one vertex per station (stations are identified by their Chinese
name across lines), one edge per pair of stations adjacent on a line,
weighted by the scheduled run time between them. Transfers are free.

Inputs:  a train_schedule_all JSON from BoyInTheSun/beijing-subway-schedule
         ({line: {direction: {train_id: [[station, "HH:MM"], ...]}}})
         the hand-curated line table data/manual/lines.csv, and
         data/manual/loop_closures.csv for the one segment of each loop line
         that no train in the snapshot runs across.
Outputs: stations.csv, lines.csv, line_stations.csv, edges_directed.csv,
         edges.csv and build_report.md in the output directory.

Only the standard library is used, and every output is sorted, so the same
inputs always produce byte-identical outputs.
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

# A stop whose time looks like "(07:42" is passed without stopping (Line 6
# overtaking trains); one like "07:40-" is the arrival half of a stop where the
# train waits to be overtaken, and is followed by a second entry for the same
# station holding the departure time.
PASS_PREFIX = "("
HOLD_SUFFIX = "-"


def parse_minutes(text):
    hh, mm = text.strip(PASS_PREFIX + HOLD_SUFFIX).split(":")
    return int(hh) * 60 + int(mm)


def train_stops(raw_stops):
    """Turn one train's raw entries into stops [(station, arr, dep, passed)].

    Minutes are monotonic within a train; times after midnight get +24h.
    """
    stops = []
    offset = 0
    last = None
    for station, text in raw_stops:
        minute = parse_minutes(text) + offset
        if last is not None and minute < last - 12 * 60:
            offset += 24 * 60
            minute += 24 * 60
        last = minute
        passed = text.startswith(PASS_PREFIX)
        if stops and stops[-1][0] == station and not passed:
            # Second half of a hold: keep the arrival, take the departure.
            stops[-1] = (station, stops[-1][1], minute, False)
        else:
            stops.append((station, minute, minute, passed))
    return stops


def read_lines(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {row["line_name_zh"]: row for row in csv.DictReader(f)}


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def fmt(x):
    return f"{x:.2f}"


def read_closures(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build(schedule_path, lines_path, closures_path, out_dir):
    schedule = json.loads(Path(schedule_path).read_text(encoding="utf-8"))
    line_meta = read_lines(lines_path)
    unknown = sorted(set(schedule) - set(line_meta))
    if unknown:
        raise SystemExit(f"lines missing from {lines_path}: {unknown}")

    # (line_id, from, to) -> list of run minutes, from stop-to-stop hops
    runs = defaultdict(list)
    # (line_id, direction, station) -> [first dep, last dep, n trains]
    service = {}
    passed_count = defaultdict(int)
    hold_count = defaultdict(int)
    line_stations = defaultdict(set)
    n_trains = defaultdict(int)

    for line_zh, directions in schedule.items():
        line_id = line_meta[line_zh]["line_id"]
        for direction, trains in directions.items():
            for raw in trains.values():
                n_trains[line_id] += 1
                hold_count[line_id] += sum(t.endswith(HOLD_SUFFIX) for _, t in raw)
                stops = train_stops(raw)
                for station, _, dep, passed in stops:
                    if passed:
                        passed_count[line_id] += 1
                        continue
                    line_stations[line_id].add(station)
                    key = (line_id, direction, station)
                    first, last, n = service.get(key, (dep, dep, 0))
                    service[key] = (min(first, dep), max(last, dep), n + 1)
                # Hops between consecutive stopping entries. A hop next to a
                # passed station is skipped: it would either bridge over the
                # passed station or time a train that does not stop.
                for a, b in zip(stops, stops[1:]):
                    if a[3] or b[3]:
                        continue
                    runs[(line_id, a[0], b[0])].append(b[1] - a[2])

    # Drop hops that skip a station some other train on the line stops at:
    # (u, w) is an express hop if u -> v -> w also exists on that line.
    succ = defaultdict(set)
    for line_id, u, w in runs:
        succ[(line_id, u)].add(w)
    express = sorted(
        (line_id, u, w)
        for line_id, u, w in runs
        if any(w in succ[(line_id, v)] for v in succ[(line_id, u)] if v != w)
    )
    for key in express:
        del runs[key]

    # Loop closures come from a manual table, timed in both directions. They
    # are kept apart from runs so they never count as observed trains.
    manual = {}
    for row in read_closures(closures_path):
        for u, w in ((row["u_zh"], row["v_zh"]), (row["v_zh"], row["u_zh"])):
            if (row["line_id"], u, w) in runs:
                raise SystemExit(f"loop closure {row} is already in the schedule")
            manual[(row["line_id"], u, w)] = float(row["minutes"])

    stations = sorted(set().union(*line_stations.values()))
    station_id = {name: f"S{i + 1:03d}" for i, name in enumerate(stations)}
    lines_by_id = {m["line_id"]: m for m in line_meta.values()}
    line_order = [m["line_id"] for m in line_meta.values() if m["line_name_zh"] in schedule]
    station_lines = defaultdict(set)
    for line_id, names in line_stations.items():
        for name in names:
            station_lines[name].add(line_id)

    def ordered(ids):
        return sorted(ids, key=line_order.index)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    write_csv(
        out / "stations.csv",
        ["station_id", "name_zh", "lines", "n_lines", "on_network_fare"],
        [
            [
                station_id[name],
                name,
                " ".join(ordered(station_lines[name])),
                len(station_lines[name]),
                int(any(lines_by_id[l]["fare_group"] == "network" for l in station_lines[name])),
            ]
            for name in stations
        ],
    )

    write_csv(
        out / "lines.csv",
        ["line_id", "name_zh", "name_en", "topology", "fare_group", "fare_source", "n_stations", "n_trains"],
        [
            [
                l,
                lines_by_id[l]["line_name_zh"],
                lines_by_id[l]["line_name_en"],
                lines_by_id[l]["topology"],
                lines_by_id[l]["fare_group"],
                lines_by_id[l]["fare_source"],
                len(line_stations[l]),
                n_trains[l],
            ]
            for l in line_order
        ],
    )

    write_csv(
        out / "line_stations.csv",
        ["line_id", "direction", "station_id", "name_zh", "first_departure", "last_departure", "n_trains"],
        [
            [l, d, station_id[s], s, hhmm(first), hhmm(last), n]
            for (l, d, s), (first, last, n) in sorted(
                service.items(), key=lambda kv: (line_order.index(kv[0][0]), kv[0][1], station_id[kv[0][2]])
            )
        ],
    )

    directed = []
    for (l, u, w), mins in runs.items():
        directed.append(
            (
                line_order.index(l), station_id[u], station_id[w],
                [l, station_id[u], station_id[w], u, w, len(mins),
                 fmt(statistics.fmean(mins)), fmt(statistics.median(mins)), fmt(min(mins)), fmt(max(mins)),
                 "schedule"],
            )
        )
    for (l, u, w), minutes in manual.items():
        directed.append(
            (
                line_order.index(l), station_id[u], station_id[w],
                [l, station_id[u], station_id[w], u, w, 0,
                 fmt(minutes), fmt(minutes), fmt(minutes), fmt(minutes), "manual"],
            )
        )
    directed.sort(key=lambda r: r[:3])
    write_csv(
        out / "edges_directed.csv",
        ["line_id", "from_id", "to_id", "from_zh", "to_zh", "n_trains",
         "mean_min", "median_min", "min_min", "max_min", "source"],
        [r[3] for r in directed],
    )

    # Undirected edges: one per (line, station pair). Weight is the mean of
    # the two directions' mean run times (or the one direction we have).
    pairs = defaultdict(dict)
    for (l, u, w), mins in runs.items():
        a, b = sorted((u, w), key=station_id.get)
        pairs[(l, a, b)][(u, w)] = statistics.fmean(mins)
    for (l, u, w), minutes in manual.items():
        a, b = sorted((u, w), key=station_id.get)
        pairs[(l, a, b)][(u, w)] = minutes
    undirected = []
    for (l, a, b), by_dir in pairs.items():
        ab, ba = by_dir.get((a, b)), by_dir.get((b, a))
        present = [x for x in (ab, ba) if x is not None]
        undirected.append(
            (
                line_order.index(l), station_id[a], station_id[b],
                [l, station_id[a], station_id[b], a, b,
                 fmt(statistics.fmean(present)),
                 "" if ab is None else fmt(ab), "" if ba is None else fmt(ba),
                 lines_by_id[l]["fare_group"],
                 "manual" if (l, a, b) in manual else "schedule"],
            )
        )
    undirected.sort(key=lambda r: r[:3])
    write_csv(
        out / "edges.csv",
        ["line_id", "u_id", "v_id", "u_zh", "v_zh", "weight_min",
         "uv_mean_min", "vu_mean_min", "fare_group", "source"],
        [r[3] for r in undirected],
    )

    report = check(line_order, lines_by_id, line_stations, pairs, express, manual,
                   passed_count, hold_count, stations, station_lines, runs, schedule_path)
    (out / "build_report.md").write_text(report, encoding="utf-8")


def hhmm(minutes):
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def check(line_order, lines_by_id, line_stations, pairs, express, manual,
          passed_count, hold_count, stations, station_lines, runs, schedule_path):
    """Sanity-check each line's shape and summarise the build as Markdown."""
    out = [f"# Model A build report\n\nSource: `{Path(schedule_path).as_posix()}`\n"]
    problems = []
    out.append("## Lines\n")
    out.append("| line | name | stations | edges | shape | expected | one-way edges | passes | holds |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for l in line_order:
        adj = defaultdict(set)
        one_way = 0
        n_edges = 0
        for (pl, a, b), by_dir in pairs.items():
            if pl != l:
                continue
            n_edges += 1
            adj[a].add(b)
            adj[b].add(a)
            one_way += len(by_dir) == 1
        degrees = sorted(len(v) for v in adj.values())
        n = len(line_stations[l])
        connected = component_size(adj) == n
        if connected and n_edges == n - 1 and degrees[-1] <= 2:
            shape = "path"
        elif connected and n_edges == n and degrees[0] == degrees[-1] == 2:
            shape = "loop"
        elif connected and n_edges == n and degrees.count(1) == 1 and degrees.count(3) == 1:
            # A tail ending in a one-way loop, like the Capital Airport Express.
            shape = "lasso"
        else:
            shape = f"other (connected={connected}, max degree={degrees[-1]})"
        expected = lines_by_id[l]["topology"]
        if shape != expected:
            problems.append(f"{l}: shape is {shape}, expected {expected}")
        if one_way and shape != "lasso":
            problems.append(f"{l}: {one_way} edge(s) timed in one direction only")
        out.append(f"| {l} | {lines_by_id[l]['line_name_zh']} | {n} | {n_edges} | {shape} | "
                   f"{expected} | {one_way} | {passed_count[l]} | {hold_count[l]} |")

    network = [s for s in stations if any(lines_by_id[l]["fare_group"] == "network" for l in station_lines[s])]
    net_adj = defaultdict(set)
    for (l, a, b) in pairs:
        if lines_by_id[l]["fare_group"] == "network":
            net_adj[a].add(b)
            net_adj[b].add(a)
    out.append("\n## Totals\n")
    out.append(f"- Stations: {len(stations)} distinct names "
               f"({sum(len(v) for v in line_stations.values())} counted per line)")
    out.append(f"- Stations reachable on the network fare: {len(network)}; "
               f"connected as one component: {component_size(net_adj) == len(network)}")
    out.append(f"- Only on separately fared lines: "
               f"{', '.join(s for s in stations if s not in network)}")
    out.append(f"- Transfer stations (2+ lines): {sum(len(station_lines[s]) > 1 for s in stations)}")
    out.append(f"- Undirected line edges: {len(pairs)}, of which "
               f"{sum(1 for k in pairs if (k[0], k[1], k[2]) in manual)} from loop_closures.csv")

    spread = sorted(
        ((max(m) - min(m), l, u, w, min(m), max(m), len(m)) for (l, u, w), m in runs.items()),
        reverse=True,
    )[:10]
    out.append("\n## Widest run-time spreads (directed, minutes)\n")
    out.append("| line | from | to | min | max | trains |")
    out.append("|---|---|---|---|---|---|")
    for _, l, u, w, lo, hi, n in spread:
        out.append(f"| {l} | {u} | {w} | {lo} | {hi} | {n} |")

    out.append("\n## Express hops removed\n")
    out.extend(f"- {l}: {u} → {w}" for l, u, w in express) if express else out.append("- none")
    out.append("\n## Problems\n")
    out.extend(f"- {p}" for p in problems) if problems else out.append("- none")
    return "\n".join(out) + "\n"


def component_size(adj):
    if not adj:
        return 0
    start = min(adj)
    seen = {start}
    stack = [start]
    while stack:
        for nxt in adj[stack.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--lines", required=True)
    parser.add_argument("--loop-closures", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    build(args.schedule, args.lines, args.loop_closures, args.out)

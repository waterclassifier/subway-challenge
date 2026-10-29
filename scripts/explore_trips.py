"""Flatten a train schedule snapshot into tidy trip and stop tables.

This is the first step of the exploration branch of the pipeline: every
explore_* analysis reads these two tables instead of the raw JSON.

Inputs:  a train_schedule_all JSON from BoyInTheSun/beijing-subway-schedule,
         data/manual/lines.csv and data/manual/loop_closures.csv.
Outputs: trips.csv  one row per trip (what the source calls a "train"),
         stops.csv  one row per trip and station, in running order.

Times. The source is built from station timetables, which it calls arrival
timetables (到站时刻表), so a stop has one time, read as the arrival. The one
exception is a Line 6 train held for an overtake, which has an arrival and a
departure; hold_min is the difference. All times are minutes after midnight
of the service day, so a stop at 00:20 the next morning is 1460. A trip runs
from its time at the first stop (start_min) to its time at the last (end_min).

Each trip is labelled with
- kind: "full" if it runs between the two termini of its direction, "short"
  if it starts or ends anywhere else, and "loop" on the loop lines (2 and 10),
  where the source cuts every lap into its own trip;
- pattern: "origin→destination", plus "express" when it passes stations;
- block: trips run back to back by one train, an estimate (see chain_blocks).

Standard library only; output is sorted and byte-identical across runs.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from build_model_a import read_closures, read_lines, train_stops, write_csv

# Trips starting before this hour belong to the previous service day.
DAY_START = 3 * 60
# A train that ends a trip continues as the next trip leaving the same place
# within this many minutes; after a longer gap we assume it went to a depot.
MAX_LAYOVER = 30
# The loop-closing run times are whole minutes too, so a lap may start up to
# this much before the previous lap "arrives" across the closing segment.
LOOP_SLACK = 2


def chain_blocks(trips, closures):
    """Group trips into blocks, one per train working, first in first out.

    A trip continues a trip of the same line that ended where it starts, at
    most MAX_LAYOVER minutes earlier, taking the train that has waited
    longest. On a loop line a lap ends one closing segment short of where the
    next lap starts (积水潭/西直门, 成寿寺/宋家庄), so a lap continues the
    same-direction lap that ended across that segment, plus its run time.
    The Capital Airport Express is not chained correctly: its outbound trips
    end at 2号航站楼 after 3号航站楼, while inbound ones start at 3号航站楼.
    Returns {trip_key: (block_id, layover before this trip or None)}.
    """
    across = {}
    for c in closures:
        across[(c["line_id"], c["u_zh"])] = (c["v_zh"], float(c["minutes"]))
        across[(c["line_id"], c["v_zh"])] = (c["u_zh"], float(c["minutes"]))
    waiting = defaultdict(list)  # place -> [(ready_min, block_id)], oldest first
    result = {}
    n_blocks = Counter()
    events = []
    for key, t in trips.items():
        events.append((t["end"], 0, key))    # arrivals before departures at a tie
        events.append((t["start"], 1, key))
    for minute, is_start, key in sorted(events):
        t = trips[key]
        line = t["line_id"]
        if is_start:
            place = (line, t["direction"] if t["loop"] else "", t["origin"])
            queue = waiting[place]
            while queue and minute - queue[0][0] > MAX_LAYOVER:
                queue.pop(0)  # that train went to a depot
            if queue and queue[0][0] <= minute + (LOOP_SLACK if t["loop"] else 0):
                ready, block = queue.pop(0)
                result[key] = (block, max(0, minute - ready))
            else:
                n_blocks[line] += 1
                result[key] = (f"{line}-{n_blocks[line]:03d}", None)
        else:
            if t["loop"] and (line, t["destination"]) in across:
                station, extra = across[(line, t["destination"])]
                place, ready = (line, t["direction"], station), minute + extra
            else:
                place, ready = (line, t["direction"] if t["loop"] else "", t["destination"]), minute
            waiting[place].append((ready, result[key][0]))
            waiting[place].sort()
    return result


def flatten(schedule_path, lines_path, closures_path, out_dir):
    schedule = json.loads(Path(schedule_path).read_text(encoding="utf-8"))
    line_meta = read_lines(lines_path)
    trip_rows, stop_rows = [], []
    trips = {}
    for line_zh, directions in schedule.items():
        meta = line_meta[line_zh]
        line_id = meta["line_id"]
        for direction, raw_trips in directions.items():
            parsed = {}
            for trip_id, raw in raw_trips.items():
                stops = train_stops(raw)
                if stops[0][1] < DAY_START:
                    stops = [(s, a + 24 * 60, d + 24 * 60, p) for s, a, d, p in stops]
                parsed[trip_id] = stops
            # The termini of a direction are the ends of its longest trip.
            longest = max(parsed.values(), key=len)
            ends = (longest[0][0], longest[-1][0])
            for trip_id, stops in sorted(parsed.items()):
                passed = sum(p for *_, p in stops)
                if meta["topology"] == "loop":
                    kind = "loop"
                elif (stops[0][0], stops[-1][0]) == ends:
                    kind = "full"
                else:
                    kind = "short"
                trips[(line_id, direction, trip_id)] = {
                    "line_id": line_id, "direction": direction, "loop": kind == "loop",
                    "origin": stops[0][0], "destination": stops[-1][0],
                    "start": stops[0][1], "end": stops[-1][1],
                    "row": [line_id, direction, trip_id, kind,
                            f"{stops[0][0]}→{stops[-1][0]}" + (" express" if passed else ""),
                            int(passed > 0), stops[0][0], stops[-1][0], len(stops) - passed, passed,
                            stops[0][1], stops[-1][1], stops[-1][1] - stops[0][1]],
                }
                for seq, (station, arr, dep, p) in enumerate(stops):
                    stop_rows.append([line_id, direction, trip_id, seq, station, arr, dep - arr, int(p)])

    blocks = chain_blocks(trips, read_closures(closures_path))
    for key, t in trips.items():
        block, layover = blocks[key]
        trip_rows.append(t["row"] + [block, "" if layover is None else f"{layover:g}"])

    order = {m["line_id"]: i for i, m in enumerate(line_meta.values())}
    trip_rows.sort(key=lambda r: (order[r[0]], r[1], r[2]))
    stop_rows.sort(key=lambda r: (order[r[0]], r[1], r[2], r[3]))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(
        out / "trips.csv",
        ["line_id", "direction", "trip_id", "kind", "pattern", "express", "origin", "destination",
         "n_stops", "n_passed", "start_min", "end_min", "duration_min", "block", "layover_before_min"],
        trip_rows,
    )
    write_csv(
        out / "stops.csv",
        ["line_id", "direction", "trip_id", "seq", "station", "time_min", "hold_min", "passed"],
        stop_rows,
    )
    kinds = Counter(r[3] for r in trip_rows)
    print(f"{len(trip_rows)} trips ({dict(sorted(kinds.items()))}) in "
          f"{len({r[13] for r in trip_rows})} blocks, {len(stop_rows)} stops")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--lines", required=True)
    parser.add_argument("--loop-closures", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    flatten(args.schedule, args.lines, args.loop_closures, args.out)

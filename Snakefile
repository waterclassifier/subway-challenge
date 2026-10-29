import csv

MANIFEST = "data/raw/MANIFEST.tsv"
MODEL_A = "data/processed/model_a"
MODEL_A_FILES = [
    "stations.csv",
    "lines.csv",
    "line_stations.csv",
    "edges_directed.csv",
    "edges.csv",
    "build_report.md",
]
EXPLORE = "data/processed/explore"
# One folder per question; the order is the order of sections in the report.
EXPLORE_STEPS = ["service", "headways", "short_turns", "express", "runtimes"]


def manifest_rows(source):
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f, delimiter="\t") if r["path"].startswith(source + "/")]


def raw_files(source):
    return [f"data/raw/{r['path']}" for r in manifest_rows(source)]


def pins(source):
    # Passed as params so that changing a commit or checksum in the manifest
    # re-runs the fetch, while merely touching the manifest does not.
    return sorted((r["path"], r["source_commit"], r["sha256"]) for r in manifest_rows(source))


rule all:
    input:
        raw_files("beijing-subway-schedule"),
        raw_files("beijing-subway-tools"),
        raw_files("noto-cjk"),
        expand(f"{MODEL_A}/{{name}}", name=MODEL_A_FILES),
        f"{EXPLORE}/report.md",


rule fetch_beijing_subway_schedule:
    output:
        raw_files("beijing-subway-schedule"),
    params:
        pins=pins("beijing-subway-schedule"),
    shell:
        "python scripts/fetch_raw.py --manifest {MANIFEST} --source beijing-subway-schedule --root data/raw"


rule fetch_beijing_subway_tools:
    output:
        raw_files("beijing-subway-tools"),
    params:
        pins=pins("beijing-subway-tools"),
    shell:
        "python scripts/fetch_raw.py --manifest {MANIFEST} --source beijing-subway-tools --root data/raw"


rule fetch_noto_cjk:
    output:
        raw_files("noto-cjk"),
    params:
        pins=pins("noto-cjk"),
    shell:
        "python scripts/fetch_raw.py --manifest {MANIFEST} --source noto-cjk --root data/raw"


rule build_model_a:
    input:
        script="scripts/build_model_a.py",
        schedule="data/raw/beijing-subway-schedule/schedule_weekday.json",
        lines="data/manual/lines.csv",
        loop_closures="data/manual/loop_closures.csv",
    output:
        expand(f"{MODEL_A}/{{name}}", name=MODEL_A_FILES),
    shell:
        "python {input.script} --schedule {input.schedule} --lines {input.lines}"
        " --loop-closures {input.loop_closures} --out {MODEL_A}"


# Exploration branch
# -----------------
# Looks at the weekday timetable itself (trains en route, frequency, short
# turns, expresses, run times), independent of any model. `snakemake -c4
# explore` builds just this branch; the report is data/processed/explore/report.md.
# Each step writes a folder of CSVs, PNGs and a summary.md; explore_report
# stitches the summaries together.


rule explore:
    input:
        f"{EXPLORE}/report.md",


rule explore_trips:
    input:
        script="scripts/explore_trips.py",
        common="scripts/build_model_a.py",
        schedule="data/raw/beijing-subway-schedule/schedule_weekday.json",
        lines="data/manual/lines.csv",
        loop_closures="data/manual/loop_closures.csv",
    output:
        f"{EXPLORE}/trips.csv",
        f"{EXPLORE}/stops.csv",
    shell:
        "python {input.script} --schedule {input.schedule} --lines {input.lines}"
        " --loop-closures {input.loop_closures} --out {EXPLORE}"


EXPLORE_INPUTS = {
    "common": "scripts/explore_common.py",
    "fonts": raw_files("noto-cjk"),
    "trips": f"{EXPLORE}/trips.csv",
    "stops": f"{EXPLORE}/stops.csv",
    "lines": "data/manual/lines.csv",
}
# Steps that draw on the subway map also need these.
MAP_INPUTS = {
    "map_script": "scripts/explore_map.py",
    "map": "data/raw/beijing-subway-tools/maps/official.json5",
    "loop_closures": "data/manual/loop_closures.csv",
}
MAP_ARGS = "--map {input.map} --loop-closures {input.loop_closures}"


rule explore_service:
    input:
        script="scripts/explore_service.py",
        **EXPLORE_INPUTS,
        **MAP_INPUTS,
    output:
        directory(f"{EXPLORE}/service"),
    shell:
        "python {input.script} --explore {EXPLORE} --lines {input.lines} " + MAP_ARGS + " --out {output}"


rule explore_headways:
    input:
        script="scripts/explore_headways.py",
        **EXPLORE_INPUTS,
        **MAP_INPUTS,
    output:
        directory(f"{EXPLORE}/headways"),
    shell:
        "python {input.script} --explore {EXPLORE} --lines {input.lines} " + MAP_ARGS + " --out {output}"


rule explore_short_turns:
    input:
        script="scripts/explore_short_turns.py",
        **EXPLORE_INPUTS,
        **MAP_INPUTS,
    output:
        directory(f"{EXPLORE}/short_turns"),
    shell:
        "python {input.script} --explore {EXPLORE} --lines {input.lines} " + MAP_ARGS + " --out {output}"


rule explore_express:
    input:
        script="scripts/explore_express.py",
        tools_line="data/raw/beijing-subway-tools/line6.json5",
        **EXPLORE_INPUTS,
    output:
        directory(f"{EXPLORE}/express"),
    shell:
        "python {input.script} --explore {EXPLORE} --tools-line {input.tools_line} --out {output}"


rule explore_runtimes:
    input:
        script="scripts/explore_runtimes.py",
        **EXPLORE_INPUTS,
    output:
        directory(f"{EXPLORE}/runtimes"),
    shell:
        "python {input.script} --explore {EXPLORE} --lines {input.lines} --out {output}"


rule explore_report:
    input:
        script="scripts/explore_report.py",
        steps=expand(f"{EXPLORE}/{{step}}", step=EXPLORE_STEPS),
    output:
        f"{EXPLORE}/report.md",
    params:
        sections=[f"{EXPLORE}/{step}/summary.md" for step in EXPLORE_STEPS],
    shell:
        "python {input.script} --schedule data/raw/beijing-subway-schedule/schedule_weekday.json"
        " --out {output} {params.sections}"

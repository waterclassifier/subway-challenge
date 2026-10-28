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
        expand(f"{MODEL_A}/{{name}}", name=MODEL_A_FILES),


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

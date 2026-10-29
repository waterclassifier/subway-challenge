# subway-challenge

Planning a route that visits as many Beijing Subway stations as possible in one day without leaving the metro system (buying another ticket at a transfer is allowed).

## Running the pipeline

The pipeline is a [Snakemake](https://snakemake.readthedocs.io/) workflow. Create the pinned environment once, then run it from the repo root:

```
conda env create -f environment.yml
conda activate subway-challenge
snakemake -c1
```

Snakemake only reruns steps whose inputs, code or pinned sources changed. `snakemake -c1 -n` shows what would run without running it.

| Rule | Output |
|---|---|
| `fetch_beijing_subway_schedule` | `data/raw/beijing-subway-schedule/`, pinned train schedules |
| `fetch_beijing_subway_tools` | `data/raw/beijing-subway-tools/`, pinned timetables, transfer data and map coordinates |
| `fetch_noto_cjk` | `data/raw/noto-cjk/`, the pinned font for figures |
| `build_model_a` | `data/processed/model_a/`, the model A station graph |
| `explore_*` | `data/processed/explore/`, a model-agnostic look at the weekday timetable; read `report.md` there |

`snakemake -c4 explore` builds only the exploration branch. `scripts/explore_map.py` draws per-line or per-segment values on the subway map and can be reused by other steps (see its docstring).

See [data/README.md](data/README.md) for sources, licenses and file formats.

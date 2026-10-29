"""Shared helpers for the explore_* analyses: loading, time axes, figure style."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# Station names are Chinese, and which CJK fonts matplotlib can find differs
# per machine, so every figure uses this pinned font (a raw input, fetched by
# the fetch_noto_cjk rule) and looks the same everywhere. Static Regular and
# Bold files, because matplotlib can't pick a weight from a variable font.
FONT_DIR = Path(__file__).resolve().parents[1] / "data/raw/noto-cjk"
FONT = FONT_DIR / "NotoSansSC-Regular.otf"
for _weight in ("Regular", "Bold"):
    font_manager.fontManager.addfont(FONT_DIR / f"NotoSansSC-{_weight}.otf")

# Chart ink and a fixed categorical order (checked for colour-vision
# deficiency in adjacent pairs).
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ("#256abf", "#f0efec", "#d03b3b")

# Rush hours as Beijing Subway uses them for fares and headways.
PERIODS = [
    ("early", 0, 7 * 60),
    ("am_peak", 7 * 60, 9 * 60),
    ("midday", 9 * 60, 17 * 60),
    ("pm_peak", 17 * 60, 19 * 60),
    ("evening", 19 * 60, 30 * 60),
]
PERIOD_ORDER = [p for p, *_ in PERIODS]

plt.rcParams.update({
    "font.sans-serif": [font_manager.FontProperties(fname=FONT).get_name()],
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "legend.frameon": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.dpi": 150,
    "lines.linewidth": 1.5,
    "axes.unicode_minus": False,
})


def load(explore_dir):
    d = Path(explore_dir)
    trips = pd.read_csv(d / "trips.csv", dtype={"trip_id": str})
    stops = pd.read_csv(d / "stops.csv", dtype={"trip_id": str})
    return trips, stops


def line_order(lines_csv):
    return list(pd.read_csv(lines_csv)["line_id"])


def period_of(minutes):
    """Rush-hour period of a time (or Series of times) in service-day minutes."""
    bins = [s for _, s, _ in PERIODS] + [PERIODS[-1][2]]
    return pd.cut(minutes, bins=bins, labels=PERIOD_ORDER, right=False)


def hhmm(minutes):
    m = int(round(minutes))
    return f"{m // 60:02d}:{m % 60:02d}"


def time_axis(ax, start=5 * 60, end=25 * 60, step=120, axis="x"):
    ticks = list(range(start, end + 1, step))
    labels = [hhmm(t) for t in ticks]
    if axis == "x":
        ax.set_xlim(start, end)
        ax.set_xticks(ticks, labels)
    else:
        ax.set_ylim(start, end)
        ax.set_yticks(ticks, labels)


def save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Fixed metadata so reruns produce identical PNGs.
    fig.savefig(path, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def md_table(df, floatfmt="{:.1f}"):
    """Render a DataFrame as a GitHub Markdown table."""
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)

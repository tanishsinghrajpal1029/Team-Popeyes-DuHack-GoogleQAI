"""Regenerate every slide figure from the result CSVs into figures/.

  python experiments/figures.py

Colours: the dataviz reference palette (light mode), validated with its palette checker:
categorical slots blue / orange / aqua for series, and one blue in two shades for
before -> after dumbbells. Aqua is below 3:1 contrast, so every line is direct-labelled.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import ncx2  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize import load, summarize  # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "figures")

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
BEFORE, AFTER = "#86b6ef", "#1c5cab"  # blue 250 -> blue 550

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 11, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 13, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.labelcolor": INK2,
})


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join("figures", name))


def score_tail():
    """Median achievable score vs K for a perfect state and for rms imbalance delta.

    chi^2 is noncentral chi^2_{K-1} with lambda = 10000 * mean(delta^2) (f = 1, x ~ 0), so
    the median score is K / median(chi^2).
    """
    K = np.unique(np.round(np.logspace(np.log10(2), np.log10(300), 60)).astype(int))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    # (anchor K, text offset in points, vertical alignment): placed in empty space
    spots = [(3, (8, 6), "bottom"), (6, (0, -10), "top"), (10, (0, -10), "top")]
    for color, d, label, (k_at, off, va) in zip(
        SERIES, [0.0, 0.01, 0.03], ["perfect state", "1% imbalance", "3% imbalance"], spots
    ):
        lam = 1e4 * d**2
        med = np.array([ncx2.median(k - 1, lam) if lam > 0 else ncx2.median(k - 1, 1e-12) for k in K])
        y = K / med
        ax.plot(K, y, color=color, lw=2, label=label)
        ax.annotate(label, (k_at, np.interp(k_at, K, y)), xytext=off, textcoords="offset points",
                    va=va, color=INK2, fontsize=10)
    ax.set_xscale("log")
    ax.set_xticks([2, 3, 4, 6, 10, 20, 50, 100, 300])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("K = C(N, n), number of valid bitstrings")
    ax.set_ylabel("median score, 10k shots")
    ax.set_ylim(0, 4.6)
    ax.set_title("Small-K cases are worth the most, and imbalance erases that")
    ax.legend(loc="upper right")
    _save(fig, "score_tail.png")


def _dumbbell(rows_before, rows_after, value, label_fmt, title, xlabel, name, log=False, cases=None,
              legend=("before (chain / routed B-E)", "now (tree / synthesized)")):
    before = {(r["N"], r["n"]): r for r in rows_before}
    after = {(r["N"], r["n"]): r for r in rows_after}
    keys = [k for k in (cases or sorted(after)) if k in before and k in after]
    names = [("W" if n == 1 else "D") + f"  N={N}" + ("" if n == 1 else f", n={n}") for N, n in keys]
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(keys) + 1.2))
    for i, k in enumerate(keys):
        b, a = value(before[k]), value(after[k])
        ax.plot([b, a], [i, i], color=AXIS, lw=1.2, zorder=1)
        ax.scatter([b], [i], s=64, color=BEFORE, zorder=2, edgecolor=SURFACE, linewidth=1.5)
        ax.scatter([a], [i], s=64, color=AFTER, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.annotate(label_fmt(a), (a, i), xytext=(8 if a >= b else -8, 0), textcoords="offset points",
                    va="center", ha="left" if a >= b else "right", color=INK2, fontsize=9)
    ax.set_yticks(range(len(keys)), names)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    if log:
        ax.set_xscale("log")
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.scatter([], [], s=64, color=BEFORE, label=legend[0])
    ax.scatter([], [], s=64, color=AFTER, label=legend[1])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    _save(fig, name)


def ceiling_and_cz():
    w = summarize(load(["w_tree.csv"]))
    g = [r for r in summarize(load(["general_n.csv"])) if r["n"] >= 2]  # its n=1 rows are trees
    before = [r for r in w if r["label"].startswith("chain")] + [r for r in g if r["label"].startswith("B-E")]
    after = [r for r in w if r["label"] == "w-tree-v2"] + [r for r in g if r["label"] == "synth"]
    cases = [(N, 1) for N in (2, 4, 6, 8, 10)] + [(4, 2), (5, 2), (6, 2), (6, 3)]
    # medians within Monte Carlo noise of the ceiling can exceed it slightly; show "≈100%"
    _dumbbell(before, after, lambda r: 100 * r["frac"], lambda v: "≈100%" if v >= 99.5 else f"{v:.0f}%",
              "Share of the perfect-state ceiling (median score)", "% of ceiling, simulated",
              "ceiling_fraction.png", cases=cases)
    _dumbbell(before, after, lambda r: r["cz"], lambda v: f"{v:.0f}",
              "CZ gates per circuit", "CZ count (log scale)", "cz_counts.png", log=True, cases=cases)

    # mid-size general n: routed B-E vs whole-state synthesis (N <= 8) / SCS blocks (k <= 4)
    before = [r for r in summarize(load(["mid_n_routed.csv"])) if r["n"] >= 2]
    after = [r for r in summarize(load(["mid_n.csv"])) if r["n"] >= 2]
    cases = [(7, 2), (7, 3), (8, 2), (8, 3), (8, 4), (9, 2), (9, 3), (9, 4), (10, 2), (10, 3), (10, 4)]
    _dumbbell(before, after, lambda r: 100 * r["frac"], lambda v: f"{v:.0f}%",
              "Mid-size cases: share of the perfect-state ceiling", "% of ceiling, simulated",
              "ceiling_fraction_mid.png", cases=cases,
              legend=("routed B-E", "synthesized (whole state N ≤ 8, SCS blocks beyond)"))
    _dumbbell(before, after, lambda r: r["cz"], lambda v: f"{v:.0f}",
              "Mid-size cases: CZ gates per circuit", "CZ count (log scale)",
              "cz_counts_mid.png", log=True, cases=cases,
              legend=("routed B-E", "synthesized (whole state N ≤ 8, SCS blocks beyond)"))


def tree_vs_chain():
    rows = summarize(load(["timing.csv"], noise="idle"))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for color, prefix, label in ((SERIES[1], "chain", "chain"), (SERIES[0], "tree", "tree")):
        rs = sorted((r for r in rows if r["label"].startswith(prefix)), key=lambda r: r["N"])
        N = [r["N"] for r in rs]
        x = [r["x"] for r in rs]
        ax.plot(N, x, color=color, lw=2, marker="o", ms=7, mec=SURFACE, mew=1.5, label=label)
        ax.annotate(label, (N[-1], x[-1]), xytext=(8, 0), textcoords="offset points", va="center", color=INK2)
    ax.set_xlabel("N (W state, n = 1)")
    ax.set_ylabel("X term x (infinite shots)")
    ax.set_title("Shallower tree keeps more coherence (gate + T1/Tφ idle noise)")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")
    _save(fig, "tree_vs_chain.png")


if __name__ == "__main__":
    score_tail()
    ceiling_and_cz()
    tree_vs_chain()

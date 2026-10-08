"""Figures for an evaluation result.

Drawn with matplotlib's object API, so no display or global pyplot state is needed and
the module works the same on a laptop and on a CI runner.
"""

from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from asag_eval.calibration import confidence_of, risk_coverage
from asag_eval.data import Predictions

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = "#2a78d6"
# One hue from light to dark: a larger share of a gold score is a darker cell.
SHARE_RAMP = LinearSegmentedColormap.from_list(
    "share", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)


def _axes(title: str, xlabel: str, ylabel: str, size: tuple[float, float] = (6.4, 4.4)):
    figure = Figure(figsize=size, dpi=150, facecolor=SURFACE, layout="constrained")
    ax = figure.add_subplot(facecolor=SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    return figure, ax


def plot_confusion_matrix(result: dict, path: Path) -> None:
    counts = np.array(result["confusion_matrix"])
    shares = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    labels = range(len(counts))
    figure, ax = _axes(
        f"Confusion matrix (n = {counts.sum()})", "Predicted score", "Gold score", (4.8, 4.4)
    )
    ax.imshow(shares, cmap=SHARE_RAMP, vmin=0, vmax=1)
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    # Surface-coloured gaps keep neighbouring cells apart without a frame.
    ax.set_xticks(np.arange(len(counts) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(counts) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=3)
    ax.tick_params(which="minor", length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    for row in labels:
        for column in labels:
            ax.text(
                column,
                row,
                f"{counts[row, column]}\n{shares[row, column]:.0%}",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if shares[row, column] > 0.5 else INK,
            )
    figure.savefig(path)


def plot_reliability(result: dict, path: Path) -> None:
    bins = result["confidence"]["reliability_bins"]
    ece = result["confidence"]["metrics"]["ece"]["value"]
    figure, ax = _axes(
        f"Reliability diagram (ECE = {ece:.3f})",
        "Confidence in the emitted score",
        "Share of correct scores",
    )
    width = bins[0]["upper"] - bins[0]["lower"]
    centres = [(b["lower"] + b["upper"]) / 2 for b in bins]
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(0.03, 0.08, "perfect calibration", color=MUTED, fontsize=8, rotation=33)
    ax.bar(centres, [b["accuracy"] for b in bins], width=width * 0.86, color=SERIES)
    for centre, b in zip(centres, bins, strict=True):
        ax.text(
            centre,
            b["accuracy"] + 0.02,
            f"n={b['count']}",
            ha="center",
            color=MUTED,
            fontsize=7,
            # Keeps the label readable where it crosses the diagonal.
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1},
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    figure.savefig(path)


def plot_risk_coverage(result: dict, preds: Predictions, path: Path) -> None:
    confidence = confidence_of(preds.probs, preds.y_pred)
    coverage, risk = risk_coverage(confidence, preds.y_true != preds.y_pred)
    area = result["confidence"]["metrics"]["aurc"]["value"]
    figure, ax = _axes(
        f"Risk-coverage curve (AURC = {area:.3f})",
        "Coverage: share of answers accepted without review",
        "Error rate among accepted answers",
    )
    ax.axhline(risk[-1], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(
        0.01,
        risk[-1],
        f"no review: {risk[-1]:.1%} errors",
        color=MUTED,
        fontsize=8,
        va="bottom",
    )
    ax.plot(coverage, risk, color=SERIES, linewidth=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(risk.max(), risk[-1]) * 1.25 + 0.01)
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    figure.savefig(path)


def write_figures(result: dict, preds: Predictions, out_dir: Path) -> tuple[str, ...]:
    """Save every figure the result supports; return their file names."""
    plot_confusion_matrix(result, out_dir / "confusion_matrix.png")
    if result["confidence"] is None:
        return ("confusion_matrix.png",)
    plot_reliability(result, out_dir / "reliability.png")
    plot_risk_coverage(result, preds, out_dir / "risk_coverage.png")
    return ("confusion_matrix.png", "reliability.png", "risk_coverage.png")

"""
Continual Learning R-Matrix Visualization
==========================================
A class for plotting standard diagnostic line plots derived from the
lower-triangular accuracy matrix R in General Continual Learning (GCL).

R[i][j] := accuracy on task j evaluated immediately after training on task i
           (defined only for j <= i, i.e., lower-triangular)

Author  : [Your Name]
Usage   :
    plotter = CLMatrixPlotter(style="publication")
    results = {
        "EWC":    [[0.85], [0.80, 0.82], [0.74, 0.78, 0.88]],
        "DER++":  [[0.87], [0.83, 0.84], [0.79, 0.81, 0.90]],
    }
    plotter.plot_diagonal(results, save_path="diagonal.pdf")
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
RMatrix = List[List[float]]          # lower-triangular R, row i has i+1 elements
ResultsDict = Dict[str, RMatrix]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _validate_lower_triangular(R: RMatrix) -> None:
    """Raise ValueError if R is not a valid lower-triangular matrix."""
    for i, row in enumerate(R):
        if len(row) != i + 1:
            raise ValueError(
                f"Row {i} has {len(row)} elements; expected {i + 1} "
                f"(lower-triangular matrix R[i] must have i+1 entries)."
            )


def _get_T(R: RMatrix) -> int:
    return len(R)


def _diagonal(R: RMatrix) -> np.ndarray:
    """Extract main diagonal: R[i][i] for i = 0..T-1."""
    return np.array([R[i][i] for i in range(_get_T(R))])


def _last_row(R: RMatrix) -> np.ndarray:
    """Extract last row: R[T-1][j] for j = 0..T-1."""
    return np.array(R[-1])


def _first_column(R: RMatrix) -> np.ndarray:
    """Extract first column: R[i][0] for i = 0..T-1."""
    return np.array([R[i][0] for i in range(_get_T(R))])


def _row_average(R: RMatrix) -> np.ndarray:
    """Compute row-wise mean: (1/(i+1)) * sum_j R[i][j] for i = 0..T-1."""
    return np.array([np.mean(R[i]) for i in range(_get_T(R))])


def _forgetting_per_task(R: RMatrix) -> np.ndarray:
    """
    Forgetting of task j:  F[j] = R[j][j] - R[T-1][j]
    Returns array of length T.
    Positive value => accuracy dropped (forgetting).
    Negative value => accuracy improved (backward plasticity).
    """
    T = _get_T(R)
    diag = _diagonal(R)
    last = _last_row(R)
    return diag - last


# ---------------------------------------------------------------------------
# Style presets
# ---------------------------------------------------------------------------

_STYLE_PRESETS = {
    "publication": {
        "font_family": "DejaVu Serif",
        "title_size": 14,
        "label_size": 12,
        "tick_size": 10,
        "legend_size": 10,
        "linewidth": 2.0,
        "markersize": 6,
        "grid_alpha": 0.25,
        "spine_visible": True,
    },
    "default": {
        "font_family": "DejaVu Sans",
        "title_size": 13,
        "label_size": 11,
        "tick_size": 9,
        "legend_size": 9,
        "linewidth": 1.8,
        "markersize": 5,
        "grid_alpha": 0.30,
        "spine_visible": True,
    },
    "minimal": {
        "font_family": "DejaVu Sans",
        "title_size": 12,
        "label_size": 11,
        "tick_size": 9,
        "legend_size": 9,
        "linewidth": 1.6,
        "markersize": 5,
        "grid_alpha": 0.20,
        "spine_visible": False,
    },
}

# Colorblind-friendly palette (Wong 2011)
_PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # pink
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CLMatrixPlotter:
    """
    Visualization toolkit for the lower-triangular accuracy matrix R
    used in General Continual Learning evaluation protocols.

    Parameters
    ----------
    style : str
        One of {"publication", "default", "minimal"}.
    dpi : int
        Resolution for rasterized outputs (PNG, etc.).
    """

    def __init__(self, style: str = "default", dpi: int = 150) -> None:
        if style not in _STYLE_PRESETS:
            raise ValueError(f"Unknown style '{style}'. Choose from {list(_STYLE_PRESETS)}.")
        self._s = _STYLE_PRESETS[style]
        self._dpi = dpi

    # ------------------------------------------------------------------
    # Internal rendering helpers
    # ------------------------------------------------------------------

    def _apply_style(self, ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
        s = self._s
        ax.set_title(title, fontsize=s["title_size"], fontfamily=s["font_family"],
                     fontweight="bold", pad=10)
        ax.set_xlabel(xlabel, fontsize=s["label_size"], fontfamily=s["font_family"])
        ax.set_ylabel(ylabel, fontsize=s["label_size"], fontfamily=s["font_family"])
        ax.tick_params(labelsize=s["tick_size"])
        ax.grid(True, linestyle="--", alpha=s["grid_alpha"], linewidth=0.8)
        if not s["spine_visible"]:
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    def _add_legend(self, ax: plt.Axes) -> None:
        ax.legend(
            fontsize=self._s["legend_size"],
            framealpha=0.85,
            edgecolor="#cccccc",
            loc="best",
        )

    def _finalize(
        self,
        fig: plt.Figure,
        save_path: Optional[str],
        tight: bool = True,
    ) -> plt.Figure:
        if tight:
            fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=self._dpi, bbox_inches="tight")
            print(f"[CLMatrixPlotter] Saved → {save_path}")
        return fig

    def _setup_figure(
        self, figsize: Tuple[int, int]
    ) -> Tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#fafafa")
        return fig, ax

    def _plot_lines(
        self,
        ax: plt.Axes,
        results: ResultsDict,
        extract_fn,
        x_offset: int = 0,
    ) -> None:
        """
        Generic line-plotting loop over all models in `results`.

        Parameters
        ----------
        extract_fn : callable
            Function that takes R (RMatrix) and returns a 1D np.ndarray of y-values.
        x_offset : int
            Starting index for x-axis ticks (0 or 1 depending on context).
        """
        s = self._s
        for idx, (model_name, R) in enumerate(results.items()):
            _validate_lower_triangular(R)
            y = extract_fn(R)
            x = np.arange(x_offset, x_offset + len(y))
            color = _PALETTE[idx % len(_PALETTE)]
            marker = _MARKERS[idx % len(_MARKERS)]
            ax.plot(
                x, y,
                label=model_name,
                color=color,
                marker=marker,
                linewidth=s["linewidth"],
                markersize=s["markersize"],
                markerfacecolor="white",
                markeredgewidth=1.5,
                markeredgecolor=color,
            )

    # ------------------------------------------------------------------
    # Public plotting methods
    # ------------------------------------------------------------------

    def plot_diagonal(
        self,
        results: ResultsDict,
        figsize: Tuple[int, int] = (8, 5),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot 1.1 — Intask (Online) Accuracy: R[i][i] vs. task index i.

        Measures plasticity: how well the model learns each new task
        immediately after being trained on it.

        Parameters
        ----------
        results : dict
            {model_name: R_matrix}
        figsize : tuple
        save_path : str, optional
            File path to save the figure (e.g., "plots/diagonal.pdf").

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, ax = self._setup_figure(figsize)
        self._plot_lines(ax, results, _diagonal, x_offset=0)
        # Integer x-ticks
        T_max = max(_get_T(R) for R in results.values())
        ax.set_xticks(np.arange(T_max))
        ax.set_xticklabels([f"$t_{{{i}}}$" for i in range(T_max)])
        self._apply_style(
            ax,
            title="Intask Accuracy (Diagonal of $R$)",
            xlabel="Task index $i$",
            ylabel="Accuracy $R[i][i]$",
        )
        self._add_legend(ax)
        return self._finalize(fig, save_path)

    def plot_last_row(
        self,
        results: ResultsDict,
        figsize: Tuple[int, int] = (8, 5),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot 1.2 — Final Accuracy Curve: R[T-1][j] vs. task index j.

        Shows per-task accuracy after training on the entire task sequence.
        The mean of this curve equals A_last.

        Parameters
        ----------
        results : dict
        figsize : tuple
        save_path : str, optional

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, ax = self._setup_figure(figsize)
        self._plot_lines(ax, results, _last_row, x_offset=0)
        T_max = max(_get_T(R) for R in results.values())
        ax.set_xticks(np.arange(T_max))
        ax.set_xticklabels([f"$t_{{{j}}}$" for j in range(T_max)])
        self._apply_style(
            ax,
            title="Final Accuracy Curve (Last Row of $R$)",
            xlabel="Task index $j$",
            ylabel="Accuracy $R[T{-}1][j]$",
        )
        self._add_legend(ax)
        return self._finalize(fig, save_path)

    def plot_first_column(
        self,
        results: ResultsDict,
        figsize: Tuple[int, int] = (8, 5),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot 1.3 — Retention Curve of Task 0: R[i][0] vs. training step i.

        Tracks how accuracy on the first task degrades as subsequent tasks
        are trained — a direct measure of catastrophic forgetting on task 0.

        Parameters
        ----------
        results : dict
        figsize : tuple
        save_path : str, optional

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, ax = self._setup_figure(figsize)
        self._plot_lines(ax, results, _first_column, x_offset=0)
        T_max = max(_get_T(R) for R in results.values())
        ax.set_xticks(np.arange(T_max))
        ax.set_xticklabels([f"After $t_{{{i}}}$" for i in range(T_max)],
                           rotation=15, ha="right")
        self._apply_style(
            ax,
            title="Retention Curve of Task 0 (First Column of $R$)",
            xlabel="Training step (task index $i$)",
            ylabel="Accuracy on task 0: $R[i][0]$",
        )
        self._add_legend(ax)
        return self._finalize(fig, save_path)

    def plot_row_average(
        self,
        results: ResultsDict,
        figsize: Tuple[int, int] = (8, 5),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot 2.1 — Average Accuracy Curve: mean(R[i][0..i]) vs. task index i.

        Running average accuracy over all tasks seen so far at each training
        step. The mean of this curve equals A_avg.

        Parameters
        ----------
        results : dict
        figsize : tuple
        save_path : str, optional

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, ax = self._setup_figure(figsize)
        self._plot_lines(ax, results, _row_average, x_offset=0)
        T_max = max(_get_T(R) for R in results.values())
        ax.set_xticks(np.arange(T_max))
        ax.set_xticklabels([f"$t_{{{i}}}$" for i in range(T_max)])
        self._apply_style(
            ax,
            title="Average Accuracy Curve (Row Mean of $R$)",
            xlabel="Task index $i$",
            ylabel=r"Avg accuracy $\frac{1}{i+1}\sum_{j=0}^{i} R[i][j]$",
        )
        self._add_legend(ax)
        return self._finalize(fig, save_path)

    def plot_forgetting_per_task(
        self,
        results: ResultsDict,
        figsize: Tuple[int, int] = (8, 5),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot 2.2 — Forgetting per Task: F[j] = R[j][j] - R[T-1][j] vs. task j.

        Positive values indicate forgetting; negative values indicate
        backward plasticity (rare, e.g., in DER variants).
        A horizontal dashed line at y=0 is drawn as reference.

        Parameters
        ----------
        results : dict
        figsize : tuple
        save_path : str, optional

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, ax = self._setup_figure(figsize)
        self._plot_lines(ax, results, _forgetting_per_task, x_offset=0)
        # Reference line at zero
        ax.axhline(y=0, color="#888888", linewidth=1.0, linestyle="--", zorder=0)
        T_max = max(_get_T(R) for R in results.values())
        ax.set_xticks(np.arange(T_max))
        ax.set_xticklabels([f"$t_{{{j}}}$" for j in range(T_max)])
        self._apply_style(
            ax,
            title="Forgetting per Task $F[j] = R[j][j] - R[T{-}1][j]$",
            xlabel="Task index $j$",
            ylabel="Forgetting $F[j]$",
        )
        self._add_legend(ax)
        return self._finalize(fig, save_path)

    # ------------------------------------------------------------------
    # Convenience: plot all five in a single figure
    # ------------------------------------------------------------------

    def plot_all(
        self,
        results: ResultsDict,
        figsize: Tuple[int, int] = (18, 10),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Render all five diagnostic plots in a single 2×3 figure grid.

        Parameters
        ----------
        results : dict
        figsize : tuple
        save_path : str, optional

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig = plt.figure(figsize=figsize, facecolor="white")
        fig.suptitle(
            "Continual Learning Diagnostic Plots",
            fontsize=self._s["title_size"] + 2,
            fontfamily=self._s["font_family"],
            fontweight="bold",
            y=1.01,
        )

        specs = [
            ("Intask Accuracy\n(Diagonal)", _diagonal,
             "Task index $i$", "Accuracy $R[i][i]$"),
            ("Final Accuracy Curve\n(Last Row)", _last_row,
             "Task index $j$", "Accuracy $R[T{-}1][j]$"),
            ("Retention Curve — Task 0\n(First Column)", _first_column,
             "Training step $i$", "Accuracy on task 0"),
            ("Average Accuracy Curve\n(Row Mean)", _row_average,
             "Task index $i$", r"Avg accuracy"),
            ("Forgetting per Task\n$F[j]=R[j][j]-R[T{-}1][j]$", _forgetting_per_task,
             "Task index $j$", "Forgetting $F[j]$"),
        ]

        axes_positions = [
            fig.add_subplot(2, 3, pos) for pos in [1, 2, 3, 4, 5]
        ]

        T_max = max(_get_T(R) for R in results.values())

        for ax, (title, fn, xlabel, ylabel) in zip(axes_positions, specs):
            ax.set_facecolor("#fafafa")
            for idx, (model_name, R) in enumerate(results.items()):
                _validate_lower_triangular(R)
                y = fn(R)
                x = np.arange(len(y))
                color = _PALETTE[idx % len(_PALETTE)]
                marker = _MARKERS[idx % len(_MARKERS)]
                s = self._s
                ax.plot(x, y, label=model_name, color=color, marker=marker,
                        linewidth=s["linewidth"], markersize=s["markersize"],
                        markerfacecolor="white", markeredgewidth=1.5,
                        markeredgecolor=color)
            # Zero reference for forgetting
            if "Forgetting" in title:
                ax.axhline(y=0, color="#888888", linewidth=1.0,
                           linestyle="--", zorder=0)
            ax.set_xticks(np.arange(T_max))
            ax.set_xticklabels([f"$t_{{{i}}}$" for i in range(T_max)])
            self._apply_style(ax, title=title, xlabel=xlabel, ylabel=ylabel)
            ax.legend(fontsize=self._s["legend_size"] - 1,
                      framealpha=0.85, edgecolor="#cccccc", loc="best")

        # Hide unused 6th subplot slot
        fig.add_subplot(2, 3, 6).set_visible(False)

        fig.tight_layout()
        return self._finalize(fig, save_path, tight=False)


# ---------------------------------------------------------------------------
# Example usage (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Synthetic lower-triangular R matrices (T=5 tasks)
    demo_results: ResultsDict = {
        "EWC": [
            [0.85],
            [0.78, 0.83],
            [0.71, 0.76, 0.87],
            [0.65, 0.70, 0.80, 0.86],
            [0.60, 0.65, 0.75, 0.81, 0.89],
        ],
        "DER++": [
            [0.87],
            [0.82, 0.85],
            [0.78, 0.80, 0.88],
            [0.74, 0.77, 0.83, 0.87],
            [0.71, 0.73, 0.80, 0.84, 0.91],
        ],
        "ER": [
            [0.84],
            [0.75, 0.82],
            [0.68, 0.73, 0.85],
            [0.61, 0.67, 0.77, 0.84],
            [0.55, 0.61, 0.72, 0.79, 0.88],
        ],
    }

    plotter = CLMatrixPlotter(style="publication", dpi=150)

    plotter.plot_diagonal(demo_results, save_path="outputs/plot_diagonal.png")
    plotter.plot_last_row(demo_results, save_path="outputs/plot_last_row.png")
    plotter.plot_first_column(demo_results, save_path="outputs/plot_first_column.png")
    plotter.plot_row_average(demo_results, save_path="outputs/plot_row_average.png")
    plotter.plot_forgetting_per_task(demo_results, save_path="outputs/plot_forgetting.png")
    plotter.plot_all(demo_results, save_path="outputs/plot_all.png")

    plt.show()
"""
Shared matplotlib style defaults.
"""
import matplotlib.pyplot as plt

PALETTE = [
    "#3498db", "#e67e22", "#2ecc71", "#e74c3c",
    "#9b59b6", "#1abc9c", "#f39c12", "#34495e",
]


def apply_style() -> None:
    """Apply consistent publication-style defaults to all subsequent plots."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "figure.autolayout": True,
    })

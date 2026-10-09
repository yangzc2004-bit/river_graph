"""Draw the implemented environmental-memory-source model as vector artwork."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path("docs/paper/latex/figures")


def main():
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(left=.01, right=.99, bottom=.01, top=.99)
    ax.set(xlim=(0, 12), ylim=(0, 8.7))
    ax.axis("off")
    green, orange, gray = "#2F7F71", "#CE7B2E", "#737E87"

    def box(x, y, w, h, title, detail, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.07,rounding_size=.12",
            edgecolor=color, facecolor=color+"10", linewidth=1.2))
        ax.text(x+w/2, y+h*.78, title, ha="center", va="center", fontsize=9,
            color="#273C48", fontweight="bold")
        ax.text(x+w/2, y+h*.33, detail, ha="center", va="center", fontsize=8.2, color="#41535E")

    def arrow(a, b, dashed=False):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12,
            color="#71818A", linewidth=1.2, linestyle="--" if dashed else "-"))

    ax.text(.25, 8.3, "DOC reconstruction without local water chemistry",
        fontsize=13, fontweight="bold", color="#273C48")
    box(.25, 6.2, 2.7, 1.4, "Receiving station", "Ecology · coordinates · season\nTemperature · discharge\nDaily hydro + visibility", green)
    box(3.65, 6.45, 3.05, 1.05, "Environmental reference", "Station-hidden ExtraTrees\nFrozen source preprocessing", green)
    box(3.65, 4.35, 3.05, 1.45, "Local temporal state", "Ecological encoder · hidden 64\nObservation-aware GRU\nCausal 12-month history", orange)
    box(.25, 2.15, 2.7, 2.45, "Source experience bank", "Permitted source DOC\nStation-OOF log residuals\nEcology + daily hydro keys\n20 ecological candidates\nCurrent-availability mask", gray)
    box(3.65, 2.0, 3.05, 1.5, "Residual source attention", "Query: receiving temporal state\nKeys: source ecology + hydro\n2 heads × 32 dimensions\nZero-value prior", orange)
    box(7.55, 4.6, 3.7, 2.15, "DOC residual readout", "Local state +41 features\nSource residual ×(1+reference)\nZero-initialized correction\n"+r"$\hat y=\max(0,C+s\,\Delta_\theta)$", orange)
    box(7.55, 1.95, 3.7, 1.65, "Saved five-seed release", "Mean prediction in mg/L\nOptional source-memory fusion\nK=1/3/5 explicit DOC support\nSource-calibrated interval + width", gray)
    arrow((3.05, 6.97), (3.54, 6.97))
    arrow((2.95, 6.3), (3.54, 5.25))
    arrow((3.03, 4.25), (3.54, 4.72))
    arrow((3.03, 2.8), (3.54, 2.8))
    arrow((5.2, 4.22), (5.2, 3.6))
    ax.text(5.36, 3.94, "Q", fontsize=9, color=orange)
    ax.text(3.14, 3.0, "K,V", fontsize=9, color=gray)
    arrow((6.8, 7.0), (7.44, 6.36))
    arrow((6.8, 5.12), (7.44, 5.12))
    arrow((6.8, 2.72), (7.45, 4.72))
    arrow((9.4, 4.45), (9.4, 3.72))
    ax.text(5.65, 1.24, "Train receiver fold A hidden; donor reference excludes A+B",
        ha="center", fontsize=8.5, color="#41535E")
    ax.text(5.65, .91, "Checkpoint, fusion and support policies use source validation only",
        ha="center", fontsize=8.5, color="#41535E")
    ax.text(.25, .45, "K0: no receiving DOC, pH or conductivity. Source observations remain available.",
        fontsize=8.3, color="#65747E")
    ax.text(.25, .14, "Source attention expresses ecological/hydrological similarity. Neural river-edge messages are absent.",
        fontsize=7.8, color="#65747E")
    ROOT.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(ROOT/f"doc_current_source_architecture_v2.{extension}", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Draw the implemented unmonitored-station DOC model as a vector schematic."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path("docs/paper/latex/figures")


def main():
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    fig.subplots_adjust(left=.01, right=.99, bottom=.01, top=.99)
    ax.set(xlim=(0, 11), ylim=(0, 5.7))
    ax.axis("off")

    def box(x, y, w, h, title, detail, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.08,rounding_size=.13",
            edgecolor=color, facecolor=color+"12", linewidth=1.2))
        ax.text(x+w/2, y+h*.76, title, ha="center", va="center", fontsize=8.5,
            color="#273C48", fontweight="bold")
        ax.text(x+w/2, y+h*.32, detail, ha="center", va="center", fontsize=8.1, color="#41535E")

    def arrow(a, b, dashed=False):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11,
            color="#71818A", linewidth=1.2, linestyle="--" if dashed else "-"))

    box(.3, 3.4, 2.55, 1.4, "New-station inputs",
        "Ecology · location · season\nTemperature · discharge\nDaily hydrology", "#36796C")
    box(.3, 1.3, 2.55, 1.25, "Source DOC context",
        "Visible source observations\nCalendar-aligned by month\nSaved source preprocessing", "#727B83")
    box(3.55, 3.8, 2.6, 1.05, "Environmental reference",
        "Station-hidden ExtraTrees\n47 matched input features", "#36796C")
    box(3.55, 1.75, 2.6, 1.55, "Temporal residual branch",
        "Ecological self encoder\nObservation-aware GRU\nCausal 12-month window\nDaily-hydrology readout", "#C87932")
    box(7.05, 3.45, 3.15, 1.35, "DOC reconstruction",
        r"$\hat y=\max(0,\ C+s\,\Delta_\theta)$"+"\nValidation-selected fusion\nFive-seed mean in mg/L", "#C87932")
    box(7.05, 1.4, 3.15, 1.3, "Optional local observations",
        "K=1 / 3 / 5 DOC values\nSource-selected correction\nEmpirical interval + width", "#727B83")
    arrow((2.95, 4.1), (3.42, 4.3))
    arrow((2.9, 3.7), (3.42, 2.9))
    arrow((2.95, 2.05), (3.42, 2.3))
    arrow((2.9, 2.5), (3.42, 3.95))
    arrow((6.25, 4.35), (6.92, 4.35))
    arrow((6.25, 2.7), (6.93, 3.6))
    arrow((8.6, 3.35), (8.6, 2.82))
    ax.text(4.87, .97, "Train on station-OOF errors;\nheld-station chemistry hidden",
        ha="center", fontsize=8.1, color="#41535E")
    arrow((4.88, 1.35), (4.88, 1.62), dashed=True)
    ax.text(.28, 5.28, "DOC reconstruction at unmonitored stations",
        fontsize=11, fontweight="bold", color="#273C48")
    ax.text(.3, .38, "K0: no local DOC, pH or conductivity. Neural edge messages are absent.",
        fontsize=8, color="#65747E")
    ax.text(.3, .08, "Source-memory weight is selected on validation; external deployment selects zero.",
        fontsize=7.6, color="#65747E")
    ROOT.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(ROOT/f"doc_deployment_architecture_v1.{extension}", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

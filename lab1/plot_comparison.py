import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def plot_comparison() -> None:
    root = Path(__file__).parent
    with (root / "comparison" / "octave_matlab_comparison.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        rows = sorted(csv.DictReader(file), key=lambda row: int(row["size"]))

    sizes = [int(row["size"]) for row in rows]
    errors = [float(row["max_absolute_error"]) for row in rows]
    matches = sum(int(row["matches"]) for row in rows)

    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    ax.plot(sizes, errors, color="#2563eb", linewidth=1.6, marker="o", markersize=5)
    ax.set_title("Porównanie wyników: Binét i GNU Octave", fontweight="bold", pad=14)
    ax.set_xscale("log", base=2)
    ax.set_xticks([2**k for k in range(sizes[-1].bit_length())])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlim(sizes[0] / 1.1, sizes[-1] * 1.1)
    ax.set_ylim(-max(errors) * 0.03, max(errors) * 1.2)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_xlabel(
        "Rozmiar macierzy n (n × n) — skala logarytmiczna\n"
        "Punkty: pomiary; linia łączy zmierzone wartości.", labelpad=10,
    )
    ax.set_ylabel("Maksymalny błąd bezwzględny: max |C Binét − C Octave|", labelpad=10)
    ax.text(
        0.03, 0.96,
        f"Zgodne wyniki: {matches}/{len(rows)}\natol = 10⁻¹², rtol = 10⁻¹²",
        transform=ax.transAxes, va="top",
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "boxstyle": "round,pad=0.5"},
    )
    ax.set_facecolor("#f8fafc")
    ax.grid(True, color="#cbd5e1", alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    output_dir = root / "plots"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "binet_octave_comparison.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    plot_comparison()

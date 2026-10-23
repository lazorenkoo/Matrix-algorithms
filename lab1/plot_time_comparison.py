import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def plot_time_comparison() -> None:
    root = Path(__file__).parent
    with (root / "binet_results.csv").open(newline="", encoding="utf-8") as file:
        binet_times = {int(row["size"]): float(row["time_seconds"]) for row in csv.DictReader(file)}
    with (root / "comparison" / "octave_matlab_comparison.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        rows = sorted(csv.DictReader(file), key=lambda row: int(row["size"]))

    rows = [row for row in rows if int(row["size"]) in binet_times]
    sizes = [int(row["size"]) for row in rows]
    octave_times = [float(row["time_seconds"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    ax.plot(sizes, [binet_times[n] for n in sizes], marker="o", markersize=4,
            color="#2563eb", label="Binét — wcześniejsze pomiary")
    ax.plot(sizes, octave_times, marker="s", markersize=4,
            color="#ea580c", label="GNU Octave — A * B")
    ax.set_title("Czas mnożenia: Binét i GNU Octave", fontweight="bold", pad=14)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([2**k for k in range(sizes[-1].bit_length())])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlim(sizes[0] / 1.1, sizes[-1] * 1.1)
    ax.set_xlabel(
        "Rozmiar macierzy n (n × n) — skala logarytmiczna\n"
        "Ten sam rozmiar; różne losowe pary macierzy i osobne sesje pomiarowe.",
        labelpad=10,
    )
    ax.set_ylabel("Czas wykonania [s] — skala logarytmiczna", labelpad=10)
    ax.set_facecolor("#f8fafc")
    ax.grid(True, color="#cbd5e1", alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend()
    output_dir = root / "plots"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "binet_octave_time.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    plot_time_comparison()

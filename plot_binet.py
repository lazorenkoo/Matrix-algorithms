import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def plot_results(csv_path: Path, algorithm: str = "binet") -> None:
    with csv_path.open(newline="", encoding="utf-8") as file:
        rows = sorted(csv.DictReader(file), key=lambda row: int(row["size"]))

    sizes = [int(row["size"]) for row in rows]
    times = [float(row["time_seconds"]) for row in rows]
    flops = [int(row["flops"]) for row in rows]

    output_dir = csv_path.parent / "plots"
    output_dir.mkdir(exist_ok=True)

    name = "Binéta" if algorithm == "binet" else "Strassena"
    color = "#2563eb" if algorithm == "binet" else "#ea580c"
    charts = [
        (sizes, times, f"Czas wykonania algorytmu {name}", "Czas wykonania [s]", f"{algorithm}_time.png"),
        (sizes, flops, f"Liczba operacji algorytmu {name}", "Liczba operacji zmiennoprzecinkowych", f"{algorithm}_flops.png"),
    ]

    memory_path = csv_path.with_name(f"{algorithm}_memory_results.csv")
    if memory_path.exists():
        with memory_path.open(newline="", encoding="utf-8") as file:
            memory_rows = sorted(csv.DictReader(file), key=lambda row: int(row["size"]))
        if memory_rows:
            memory_sizes = [int(row["size"]) for row in memory_rows]
            memory = [float(row["peak_traced_memory_mib"]) for row in memory_rows]
            charts.append((
                memory_sizes, memory, f"Pamięć algorytmu {name}",
                "Szczyt śledzonych alokacji [MiB]", f"{algorithm}_memory.png",
            ))

    for chart_sizes, values, title, ylabel, filename in charts:
        fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
        ax.plot(chart_sizes, values, color=color, linewidth=1.6)
        ax.scatter(chart_sizes, values, color=color, s=12, zorder=3)
        ax.set_title(title, pad=14, fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel(
            "Rozmiar macierzy n (n × n) — skala logarytmiczna\n"
            "Punkty: pomiary; linia łączy zmierzone wartości.",
            labelpad=10,
        )
        ax.set_ylabel(ylabel + " — skala logarytmiczna", labelpad=10)
        ax.set_xlim(chart_sizes[0] / 1.1, max(chart_sizes[-1], chart_sizes[0] + 1) * 1.1)
        ticks = [2**k for k in range(chart_sizes[-1].bit_length())]
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_facecolor("#f8fafc")
        ax.grid(True, which="major", color="#cbd5e1", alpha=0.65)
        ax.grid(True, which="minor", axis="y", color="#cbd5e1", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        output_path = output_dir / filename
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["binet", "strassen"], default="binet")
    args = parser.parse_args()
    plot_results(Path(__file__).with_name(f"{args.algorithm}_results.csv"), args.algorithm)

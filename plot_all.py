import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def read_results(path: Path):
    with path.open(newline="", encoding="utf-8") as file:
        return {int(row["size"]): row for row in csv.DictReader(file)}


def plot_all(compiled: bool = False) -> None:
    root = Path(__file__).parent
    output_dir = root / "plots"
    output_dir.mkdir(exist_ok=True)
    methods = [
        ("binet", "Binét", "#2563eb", "o"),
        ("strassen", "Strassen", "#ea580c", "s"),
        ("alphatensor", "AlphaTensor", "#16a34a", "^"),
    ]
    if compiled:
        methods = [(method + "_compiled", label + " (Cython)", color, marker)
                   for method, label, color, marker in methods]
    charts = [
        (
            "results", "time_seconds", "Porównanie czasu mnożenia macierzy",
            "Czas wykonania [s]", "comparison_time.png",
            "Osobne sesje; Binét: wcześniejsze losowe pary. Strassen i AlphaTensor: wspólne zapisane pary.",
        ),
        (
            "memory_results", "peak_traced_memory_mib", "Porównanie pamięci algorytmów",
            "Szczyt śledzonych alokacji [MiB]", "comparison_memory.png",
            "Tracemalloc: alokacje podczas mnożenia, bez buforów wejściowych; nie cała pamięć procesu.",
        ),
        (
            "results", "flops", "Porównanie liczby operacji",
            "Liczba operacji zmiennoprzecinkowych", "comparison_flops.png",
            "Mnożenia, dodawania i odejmowania; AlphaTensor uwzględnia obliczenia na dopełnionych zerach.",
        ),
    ]

    for suffix, field, title, ylabel, filename, note in charts:
        if compiled:
            filename = filename.replace(".png", "_compiled.png")
            if field == "time_seconds":
                note = "Wszystkie metody: skompilowane moduły Cython i wspólne zapisane pary macierzy."
        datasets = [read_results(root / f"{method}_{suffix}.csv") for method, _, _, _ in methods]
        sizes = sorted(set.intersection(*(set(dataset) for dataset in datasets)))
        fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
        for (_, label, color, marker), dataset in zip(methods, datasets):
            values = [float(dataset[n][field]) for n in sizes]
            ax.plot(sizes, values, color=color, marker=marker, markersize=5,
                    linewidth=1.6, label=label)

        ax.set_title(title, fontweight="bold", pad=14)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([2**k for k in range(sizes[-1].bit_length())])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlim(sizes[0] / 1.1, sizes[-1] * 1.1)
        ax.set_xlabel("Rozmiar macierzy n (n × n) — skala logarytmiczna", labelpad=10)
        ax.set_ylabel(ylabel + " — skala logarytmiczna", labelpad=10)
        ax.set_facecolor("#f8fafc")
        ax.grid(True, which="major", color="#cbd5e1", alpha=0.65)
        ax.grid(True, which="minor", axis="y", color="#cbd5e1", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend()
        fig.supxlabel(
            "Tylko rozmiary zmierzone dla wszystkich metod; linie łączą punkty.\n" + note,
            fontsize=8,
        )
        output_path = output_dir / filename
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        print(f"Plot saved to {output_path} ({len(sizes)} common sizes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled", action="store_true")
    args = parser.parse_args()
    plot_all(compiled=args.compiled)

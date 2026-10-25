import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def plot_results(implementation="compiled", task="inverse", results_dir=None, output_dir=None):
    root = Path(__file__).parent
    results_dir = Path(results_dir) if results_dir is not None else root / "results"
    with (results_dir / f"{task}_{implementation}.csv").open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row["matches_reference"] == "1"]
    output_dir = Path(output_dir) if output_dir is not None else root / "plots"
    output_dir.mkdir(exist_ok=True)
    for field, ylabel, filename in [
        ("time_seconds", "Czas odwracania macierzy [s]" if task == "inverse" else "Czas faktoryzacji LU [s]", "time"),
        ("flops", "Liczba operacji zmiennoprzecinkowych", "flops"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
        for method, color, marker in [("binet", "#2563eb", "o"), ("strassen", "#ea580c", "s"),
                                      ("laderman", "#16a34a", "^")]:
            values = sorted((row for row in rows if row["method"] == method and float(row[field]) > 0),
                            key=lambda row: int(row["size"]))
            if values:
                ax.plot([int(row["size"]) for row in values], [float(row[field]) for row in values],
                        color=color, marker=marker, markersize=4, linewidth=1.5)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel("Rozmiar macierzy n (n × n)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="major", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
        path = output_dir / f"{task}_{implementation}_{filename}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=["python", "compiled"], default="compiled")
    parser.add_argument("--task", choices=["inverse", "lu"], default="inverse")
    args = parser.parse_args()
    plot_results(args.implementation, args.task)

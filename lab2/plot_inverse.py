import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def plot_results(implementation="compiled"):
    root = Path(__file__).parent
    with (root / "results" / f"inverse_{implementation}.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    output_dir = root / "plots"
    output_dir.mkdir(exist_ok=True)
    for field, ylabel, filename in [
        ("time_seconds", "Czas odwracania macierzy [s]", "time"),
        ("flops", "Liczba operacji zmiennoprzecinkowych", "flops"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
        for method, color, marker in [("binet", "#2563eb", "o"), ("strassen", "#ea580c", "s")]:
            values = sorted((row for row in rows if row["method"] == method), key=lambda row: int(row["size"]))
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
        path = output_dir / f"inverse_{implementation}_{filename}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=["python", "compiled"], default="compiled")
    plot_results(parser.parse_args().implementation)

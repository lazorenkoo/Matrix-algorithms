import argparse
import csv
from pathlib import Path

import numpy as np

from binet import binet
from matrix_cases import case_directory, load_matrix_pair


def compare_binet(sizes: list[int], resume: bool = False) -> None:
    output_dir = Path(__file__).parent / "comparison"
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "numpy_comparison.csv"
    text_path = output_dir / "numpy_comparison.txt"
    header = (
        "Porównanie algorytmu Binéta z NumPy\n"
        "Ziarno generatora dla każdego rozmiaru: 42\n"
        "Typ danych: float64; atol=1e-12, rtol=1e-12\n"
        "Weryfikację w Octave/MATLAB uruchamia się osobno: compare_binet.m.\n\n"
        "Rozmiar | Maks. błąd bezwzględny | Zgodność | Mnożenia | Dodawania | FLOPs\n"
    )
    completed = set()
    if resume and csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as file:
            completed = {int(row["size"]) for row in csv.DictReader(file)}
    else:
        text_path.write_text(header, encoding="utf-8")
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow([
                "size", "max_absolute_error", "matches",
                "multiplications", "additions", "flops",
            ])

    for n in sorted(set(sizes)):
        if n in completed:
            continue
        print(f"Porównanie n={n}", flush=True)
        A, B = load_matrix_pair(n)

        C_binet, multiplications, additions = binet(A, B)
        C_numpy = A @ B
        max_error = np.max(np.abs(C_binet - C_numpy))
        matching = np.allclose(C_binet, C_numpy, rtol=1e-12, atol=1e-12)
        flops = multiplications + additions

        size_dir = case_directory(n)
        size_dir.mkdir(exist_ok=True)
        for name, matrix in [
            ("A", A), ("B", B), ("C_binet", C_binet), ("C_numpy", C_numpy),
        ]:
            np.savetxt(size_dir / f"{name}.csv", matrix, delimiter=",", fmt="%.17g")
            if n == 4:
                np.savetxt(output_dir / f"{name}.csv", matrix, delimiter=",", fmt="%.17g")

        with csv_path.open("a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow([
                n, max_error, int(matching), multiplications, additions, flops,
            ])
        line = (
            f"{n} × {n} | {max_error:.6e} | {'TAK' if matching else 'NIE'} | "
            f"{multiplications} | {additions} | {flops}\n"
        )
        with text_path.open("a", encoding="utf-8") as file:
            file.write(line)
        print(line.rstrip(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Binet results with NumPy.")
    parser.add_argument("--sizes", type=int, nargs="+")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.sizes is not None:
        sizes = args.sizes
    elif args.full:
        sizes = [1, 2, 3, 4, 5, 8, 17, 32, 64, 96, 128, 160, 224, 256, 350, 512]
    else:
        sizes = [1, 2, 3, 4, 5, 8, 17, 32, 64, 96, 128]
    compare_binet(sizes, resume=args.resume)

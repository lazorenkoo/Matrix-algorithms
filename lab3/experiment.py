import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from PIL import Image

from lab3.hmatrix import compress, reconstruct


CHANNELS = ("R", "G", "B")


def leaves(node):
    if node.children:
        for child in node.children:
            yield from leaves(child)
    else:
        yield node


def rectangle(x, y, width, height):
    return [(x, y), (x+width, y), (x+width, y+height), (x, y+height)]


def draw_structure(ax, root, color_rank=False):
    outlines, factors, dense, ranks = [], [], [], []
    for node in leaves(root):
        r0, r1, c0, c1 = node.bounds
        height, width = node.shape
        outlines.append(rectangle(c0, r0, width, height))
        if node.dense is not None:
            dense.append(rectangle(c0, r0, width, height))
        elif node.left is not None:
            rank = node.left.shape[1]
            ranks.extend((rank, rank))
            factors.extend((
                rectangle(c0, r0, rank, height),
                rectangle(c0, r0, width, rank),
            ))
    ax.add_collection(PolyCollection(dense, facecolor="#b7cad7", edgecolor="none"))
    bars = PolyCollection(factors, facecolor="#183f63", edgecolor="none")
    if color_rank:
        bars.set_facecolor(plt.colormaps["viridis"](np.asarray(ranks) / 8))
    ax.add_collection(bars)
    ax.add_collection(PolyCollection(outlines, facecolor="none", edgecolor="#555555", linewidth=.3))
    height, width = root.shape
    ax.set(xlim=(0, width), ylim=(height, 0), aspect="equal")
    ax.axis("off")


def stored_values(root):
    return sum(
        array.size
        for node in leaves(root)
        for array in (node.left, node.right, node.dense)
        if array is not None
    )


def pixels(array):
    return np.rint(np.clip(array, 0, 255)).astype(np.uint8)


def run(source, output, intermediate=False):
    output.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
    channels = tuple(rgb[:, :, index] for index in range(3))
    spectra = [np.linalg.svd(channel, compute_uv=False) for channel in channels]

    Image.fromarray(pixels(rgb)).save(output / "original_rgb.png")
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for name, color, channel, sigma in zip(CHANNELS, ("red", "green", "blue"), channels, spectra):
        Image.fromarray(pixels(channel)).save(output / f"original_{name}.png")
        ax.plot(np.arange(1, sigma.size + 1), sigma, label=name, color=color)
    ax.set(xlabel="Indeks i", ylabel="Wartość osobliwa", yscale="symlog", title="Widma kanałów RGB")
    ax.legend()
    fig.savefig(output / "spectra.png", dpi=170)
    plt.close(fig)

    rows = []
    variants = (("d10", 10), ("d30", 30), ("d90", 90)) if intermediate else (
        ("max", None), ("min", None), ("half_min", None),
    )
    for rank in (1, 4):
        for label, absolute in variants:
            key = f"r{rank}_{label}"
            print(key, flush=True)
            fig, axes = plt.subplots(2, 3, figsize=(10, 7), layout="constrained")
            fig.suptitle(f"r = {rank}, próg: {label}")
            reconstructed = []
            for col, (name, channel, sigma) in enumerate(zip(CHANNELS, channels, spectra)):
                threshold = float(absolute if absolute is not None else {
                    "max": sigma[0], "min": sigma[-1], "half_min": sigma[-1] / 2,
                }[label])
                root = compress(channel, rank, threshold)
                restored = reconstruct(root)
                channel_pixels = pixels(restored)
                reconstructed.append(channel_pixels)
                Image.fromarray(channel_pixels).save(output / f"{key}_{name}.png")
                draw_structure(axes[0, col], root)
                axes[0, col].set_title(f"{name}: struktura")
                axes[1, col].imshow(channel_pixels, cmap="gray", vmin=0, vmax=255)
                axes[1, col].set_title(f"{name}: rekonstrukcja")
                axes[1, col].axis("off")
                count = stored_values(root)
                rows.append((rank, label, name, threshold, count, count/channel.size,
                             np.sqrt(np.mean((channel-channel_pixels)**2))))
            fig.savefig(output / f"{key}_channels.png", dpi=160)
            plt.close(fig)
            Image.fromarray(np.stack(reconstructed, axis=-1)).save(output / f"{key}_rgb.png")

    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("rank", "threshold", "channel", "delta", "stored_values",
                         "stored_to_dense", "pixel_rmse"))
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--intermediate", action="store_true")
    args = parser.parse_args()
    run(args.image, args.output, args.intermediate)

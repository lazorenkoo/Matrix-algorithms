import argparse
import csv
from pathlib import Path

import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image
from scipy.ndimage import uniform_filter

from lab3.experiment import draw_structure, pixels, plt
from lab3.hmatrix import HMatrixNode, leading_svd, reconstruct


def build(matrix, bounds, max_rank, delta, minimum, lossless):
    r0, r1, c0, c1 = bounds
    block = matrix[r0:r1, c0:c1]
    height, width = block.shape
    if np.linalg.norm(block) <= 1e-12:
        return HMatrixNode(bounds)
    u, s, vt = leading_svd(block, max_rank + 1)
    rank = int(np.count_nonzero(s > (1e-10 if lossless else delta)))
    if rank <= max_rank and rank * (height + width) < block.size:
        if rank == 0:
            return HMatrixNode(bounds)
        return HMatrixNode(bounds, left=u[:, :rank].copy(),
                           right=s[:rank, None] * vt[:rank],
                           singular_values=s[:rank].copy())
    if min(block.shape) <= minimum:
        return HMatrixNode(bounds, dense=block.copy())
    rm, cm = (r0+r1)//2, (c0+c1)//2
    return HMatrixNode(bounds, children=tuple(
        build(matrix, (a,b,c,d), max_rank, delta, minimum, lossless)
        for a,b in ((r0,rm),(rm,r1)) for c,d in ((c0,cm),(cm,c1))
    ))


def statistics(node):
    if node.children:
        return sum(map(statistics, node.children))
    if node.dense is not None:
        return node.dense.size
    if node.left is not None:
        return node.left.size + node.right.size + node.singular_values.size
    return 0


def quality(a, b):
    mse = np.mean((a-b)**2)
    psnr = -10*np.log10(mse) if mse else float("inf")
    ua, ub = uniform_filter(a, (7,7,1)), uniform_filter(b, (7,7,1))
    va = (uniform_filter(a*a, (7,7,1))-ua*ua)*49/48
    vb = (uniform_filter(b*b, (7,7,1))-ub*ub)*49/48
    cov = (uniform_filter(a*b, (7,7,1))-ua*ub)*49/48
    score = ((2*ua*ub+.01**2)*(2*cov+.03**2) /
             ((ua*ua+ub*ub+.01**2)*(va+vb+.03**2)))
    return psnr, float(score[3:-3,3:-3].mean())


def run(source, output):
    output.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        original = np.asarray(image.convert("RGB"), dtype=float)/255
    if min(original.shape[:2]) < 7:
        raise ValueError("SSIM requires an image of at least 7 × 7 pixels.")
    hsv = rgb_to_hsv(original)
    symmetric = [(8,1e-7,16),(4,.05,8),(2,.25,16),(1,.1,8)]
    profiles = [[p]*3 for p in symmetric] + [
        [(4,.08,8),(2,.3,8),(6,.02,8)],
        [(1,.4,32),(6,.02,8),(6,.02,8)],
    ]
    overview, panels = plt.subplots(7,2,figsize=(8,23),layout="constrained")
    rows = []
    h,w,_ = original.shape
    for col, (space, image) in enumerate((("RGB",original),("HSV",hsv))):
        panels[0,col].imshow(original)
        panels[0,col].set_title(f"Oryginał / {space}")
        panels[0,col].axis("off")
        for index, profile in enumerate(profiles,1):
            config = profile if space == "HSV" or index <= 4 else [symmetric[1]]*3
            restored = []
            stored = 0
            fig, axes = plt.subplots(4,3,figsize=(10,12),layout="constrained")
            for k, (name, params) in enumerate(zip(space, config)):
                channel = image[:,:,k]
                root = build(channel,(0,h,0,w),*params,index == 1)
                approx = np.clip(reconstruct(root),0,1)
                restored.append(approx)
                stored += statistics(root)
                error = np.abs(channel-approx)
                for j, (array, title) in enumerate(((channel,"Oryginał"),(approx,"Rekonstrukcja"))):
                    axes[j,k].imshow(array,cmap="gray",vmin=0,vmax=1)
                    axes[j,k].set_title(f"{name}: {title}")
                    axes[j,k].axis("off")
                draw_structure(axes[2,k], root, color_rank=True)
                axes[2,k].set_title(f"{name}: bloki")
                im = axes[3,k].imshow(error,cmap="magma",vmin=0,vmax=1)
                axes[3,k].set_title(f"{name}: błąd")
                axes[3,k].axis("off")
            rank_map = plt.cm.ScalarMappable(norm=plt.Normalize(0, 8), cmap="viridis")
            fig.colorbar(rank_map, ax=axes[2,:], shrink=.65, label="Rząd", ticks=range(1,9))
            fig.colorbar(im,ax=axes[3,:],shrink=.65,label="Błąd bezwzględny")
            key=f"{space.lower()}_{index}"
            fig.savefig(output/f"{key}_channels.png",dpi=120)
            plt.close(fig)
            rgb=np.stack(restored,axis=-1)
            if space=="HSV":
                rgb=hsv_to_rgb(rgb)
            saved=pixels(255*rgb)
            Image.fromarray(saved).save(output/f"{key}.png")
            psnr, ssim=quality(original,saved.astype(float)/255)
            ratio=stored/original.size
            if index == 1:
                np.testing.assert_array_equal(saved,pixels(255*original))
            rows.append((space,index,str(config),stored,ratio,psnr,ssim))
            ax=panels[index,col]
            ax.imshow(saved)
            ax.set_title(f"{space} {index} · {ratio:.1%} · PSNR {psnr:.2f} · SSIM {ssim:.3f}")
            ax.axis("off")
            print(key,round(ratio,3),round(psnr,2),round(ssim,4),flush=True)
    overview.savefig(output/"rgb_vs_hsv.png",dpi=130)
    plt.close(overview)
    with (output/"metrics.csv").open("w",encoding="utf-8",newline="") as file:
        writer=csv.writer(file)
        writer.writerow(("space","profile","rank_delta_minimum","stored_values","stored_to_dense","psnr","ssim"))
        writer.writerows(rows)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("image",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    run(args.image,args.output)

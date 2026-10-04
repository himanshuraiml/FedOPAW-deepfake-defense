"""Architecture diagram for FedOPAW (image-only). No caption is drawn into the image."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(7.2, 4.1)); ax.set_xlim(0, 100); ax.set_ylim(0, 58); ax.axis("off")
def box(x, y, w, h, title, sub, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center", fontsize=7.2, fontweight="bold", color="#1a237e")
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=5.6, color="#37474f")
def panel(x, w, title, fc, ec):
    ax.add_patch(FancyBboxPatch((x, 1.5), w, 53, boxstyle="round,pad=0.3,rounding_size=1.5", fc=fc, ec=ec, lw=1.6, ls="--"))
    ax.text(x + w / 2, 56, title, ha="center", fontsize=8.5, fontweight="bold", color=ec)
def arrow(p, q, c="#546e7a", style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, color=c, lw=1.2, ls=ls))

panel(1, 41, "Client node k  (k = 1 ... N)", "#e8f5e9", "#2e7d32")
box(3, 40, 18, 11, "Local detector $\\theta_k$", "ResNet-18 + spectrum", "#c8e6c9", "#2e7d32")
box(22, 40, 18, 11, "Open-set module", "CLIP + freq. score", "#c8e6c9", "#2e7d32")
box(3, 26, 18, 10, "Provenance parser", "C2PA manifest $\\to \\tau$", "#c8e6c9", "#2e7d32")
box(22, 26, 18, 10, "Fed-EWC + GEM", "penalty + projection", "#b2dfdb", "#00695c")
box(3, 7, 37, 14, "Private image data $D_k$", "real and fake images, optional C2PA metadata\nnever leave the client", "#dcedc8", "#558b2f")
arrow((12, 39.6), (12, 36.6)); arrow((31, 39.6), (31, 36.6)); arrow((31, 21.4), (31, 25.6))

panel(60, 39, "Global server", "#f3e5f5", "#6a1b9a")
box(62, 40, 35, 11, "Global detector $\\theta_G$", "same architecture as the clients", "#e1bee7", "#6a1b9a")
box(62, 27, 35, 10, "FedAvg + distillation", "weighted average, then KD on $P$", "#e1bee7", "#6a1b9a")
box(62, 14.5, 17, 10, "Fisher merge", "$F_G=\\sum_k \\frac{|D_k|}{|D|}F_k$", "#e1bee7", "#6a1b9a")
box(80, 14.5, 17, 10, "AdvMark-Fed", "watermark co-design", "#e1bee7", "#6a1b9a")
box(62, 3, 35, 8, "Proxy set $P$", "unlabeled images held by the server", "#d1c4e9", "#4527a0")
arrow((79.5, 39.6), (79.5, 37.4)); arrow((72, 26.6), (70.5, 25)); arrow((88, 26.6), (88.5, 25))
arrow((79.5, 11.4), (79.5, 13.6), c="#4527a0", ls=":")

arrow((42.5, 45), (59.5, 45), c="#6a1b9a")
ax.text(51, 48.3, "$\\theta_G$ broadcast (fp32)", ha="center", fontsize=6, color="#6a1b9a")
arrow((59.5, 32), (42.5, 32), c="#2e7d32")
ax.text(51, 34.3, "$\\Delta\\theta_k$ (int8/sparse)\n$F_k$", ha="center", fontsize=6, color="#2e7d32")
ax.text(51, 19, "raw images stay\non the client;\nmessages are\nnot encrypted", ha="center", fontsize=5.8, color="#b71c1c", style="italic")
fig.savefig("../figures_ieee/fig01_architecture_v2.png", dpi=250, bbox_inches="tight"); print("ok")

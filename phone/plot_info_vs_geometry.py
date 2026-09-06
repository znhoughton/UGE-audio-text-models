#!/usr/bin/env python3
"""Money figure: Information vs Geometry across depth, per component x size.
Color = pole identity (acoustic=blue, linguistic=vermillion; CVD-safe Okabe-Ito).
Style = measure (solid=information/decodability, dashed=geometry/mutual-kNN).
Shaded band between solid & dashed = the info-geometry dissociation."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

REPO = Path("/opt/modeling/zhoughton/misc/UGE-audio-text-models")
d = json.load(open(Path(__file__).resolve().parent / "results" / "info_vs_geom_results.json"))["results"]

AC = "#0072B2"   # acoustic (kaldi) - blue
LG = "#D55E00"   # linguistic (olmo/word-id) - vermillion
INK = "#222222"; MUT = "#888888"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#cccccc",
                     "axes.linewidth": 0.8, "figure.dpi": 140})

models = ["whisper-base", "whisper-large-v2"]; comps = ["enc", "dec"]
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True, sharex=True)
for r, comp in enumerate(comps):
    for c, M in enumerate(models):
        ax = axes[r, c]; rows = d[M][comp]
        x = [q["depth"] for q in rows]
        gk = [q["geo_kaldi"] for q in rows]; ik = [q["inf_kaldi"] for q in rows]
        go = [q["geo_olmo"] for q in rows]; io = [q["inf_olmo"] for q in rows]
        ax.fill_between(x, gk, ik, color=AC, alpha=0.12, lw=0)
        ax.fill_between(x, go, io, color=LG, alpha=0.12, lw=0)
        ax.plot(x, ik, color=AC, lw=2, label="acoustic · information")
        ax.plot(x, gk, color=AC, lw=2, ls=(0, (4, 2)), label="acoustic · geometry")
        ax.plot(x, io, color=LG, lw=2, label="linguistic · information")
        ax.plot(x, go, color=LG, lw=2, ls=(0, (4, 2)), label="linguistic · geometry")
        ax.axhline(0, color="#dddddd", lw=0.8, zorder=0)
        comp_name = "Encoder" if comp == "enc" else "Decoder"
        size = M.replace("whisper-", "")
        ax.set_title(f"{comp_name}  ·  {size}", fontsize=11, color=INK, loc="left")
        ax.grid(True, axis="y", color="#eeeeee", lw=0.7)
        ax.set_xlim(0, 1); ax.set_ylim(-0.06, 0.48)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        if r == 1: ax.set_xlabel("relative depth (input → output)", color=MUT)
        if c == 0: ax.set_ylabel("alignment", color=MUT)

# one legend
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=9,
           bbox_to_anchor=(0.5, 1.03))
fig.suptitle("Information is retained where geometry is not — per layer, across scale",
             y=1.08, fontsize=13, color=INK)
fig.text(0.5, -0.02,
         "Shaded band = information − geometry gap.  Encoder: acoustic information stays high "
         "while geometry stays low (large gap).  Decoder: sheds acoustic info, keeps linguistic.",
         ha="center", color=MUT, fontsize=8.5)
plt.tight_layout()
out = Path(__file__).resolve().parent / "figures" / "info_vs_geometry_by_layer.png"
out.parent.mkdir(exist_ok=True)
plt.savefig(out, bbox_inches="tight", facecolor="white")
print("saved", out)

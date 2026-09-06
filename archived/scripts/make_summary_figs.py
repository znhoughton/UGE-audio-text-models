import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open(r"C:\Users\zhoughton\Documents\UGE-audio-text-models\DecompositionData\decomposition_results.json") as f:
    data = json.load(f)

OUT = r"C:\Users\zhoughton\Documents\UGE-audio-text-models\DecompositionPlots"

SIZE_ORDER = ["base", "small", "medium", "large-v1", "large-v2", "large"]
def sort_key(name):
    for i, s in enumerate(SIZE_ORDER):
        if s in name: return i
    return 99

enc = sorted([m for m in data["results"] if m.endswith("-enc")], key=sort_key)
dec = sorted([m for m in data["results"] if m.endswith("-dec")], key=sort_key)
models = enc + dec
short  = [m.replace("whisper-","").replace("-enc","").replace("-dec","") for m in models]
x = np.arange(len(models))
w = 0.65

# ── Figure 1: Venn partition ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5.5))

u_a, ov, u_t, nei = [], [], [], []
for m in models:
    o = data["overlap"][m]
    u_a.append(o["v_acoustic"]    - o["overlap_variance"])
    ov.append(o["overlap_variance"])
    u_t.append(o["v_textual_raw"] - o["overlap_variance"])
    nei.append(1 - o["v_joint"])

b = np.zeros(len(models))
for vals, col, lbl in zip(
    [u_a, ov, u_t, nei],
    ["#2196F3", "#9C27B0", "#4CAF50", "#BDBDBD"],
    ["Unique Acoustic", "Joint (Overlap)", "Unique Textual", "Neither"]
):
    vals = np.array(vals)
    bars = ax.bar(x, vals, bottom=b, color=col, label=lbl, width=w,
                  edgecolor="white", linewidth=0.5)
    for bar, v, bot in zip(bars, vals, b):
        if v > 0.04:
            ax.text(bar.get_x()+bar.get_width()/2, bot+v/2, f"{v:.2f}",
                    ha="center", va="center", fontsize=7.5,
                    color="white", fontweight="bold")
    b += vals

ax.axvline(len(enc)-0.5, color="black", linestyle="--", lw=1, alpha=0.4)
ax.text(len(enc)/2 - 0.5,          1.07, "Encoder", ha="center", fontsize=10,
        transform=ax.get_xaxis_transform())
ax.text(len(enc) + len(dec)/2 - 0.5, 1.07, "Decoder", ha="center", fontsize=10,
        transform=ax.get_xaxis_transform())
ax.set_xticks(x); ax.set_xticklabels(short, fontsize=9)
ax.set_ylabel("Fraction of Whisper Variance", fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_title("Variance Partition: Unique Acoustic / Joint Overlap / Unique Textual / Neither\n"
             "(Acoustic ref: kaldi-librispeech  |  Textual ref: olmo-7b  |  k=50)", fontsize=11)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT + "/fig1_venn_partition.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig1_venn_partition.png")

# ── Figure 2: Sequential decomposition (adaptive k) ──────────────────────────
fig, ax = plt.subplots(figsize=(13, 5.5))

ac, tx, inter, res = [], [], [], []
for m in models:
    r = data["results"][m]
    ac.append(r["acoustic"]); tx.append(r["textual"])
    inter.append(r["interaction"]); res.append(r["residual"])

b = np.zeros(len(models))
for vals, col, lbl in zip(
    [ac, tx, inter, res],
    ["#2196F3", "#4CAF50", "#FF9800", "#BDBDBD"],
    ["Acoustic (sequential)", "Textual (sequential)", "Interaction", "Residual"]
):
    vals = np.array(vals)
    bars = ax.bar(x, vals, bottom=b, color=col, label=lbl, width=w,
                  edgecolor="white", linewidth=0.5)
    for bar, v, bot in zip(bars, vals, b):
        if v > 0.04:
            ax.text(bar.get_x()+bar.get_width()/2, bot+v/2, f"{v:.2f}",
                    ha="center", va="center", fontsize=7.5,
                    color="white", fontweight="bold")
    b += vals

ax.axvline(len(enc)-0.5, color="black", linestyle="--", lw=1, alpha=0.4)
ax.text(len(enc)/2 - 0.5,          1.07, "Encoder", ha="center", fontsize=10,
        transform=ax.get_xaxis_transform())
ax.text(len(enc) + len(dec)/2 - 0.5, 1.07, "Decoder", ha="center", fontsize=10,
        transform=ax.get_xaxis_transform())
ax.set_xticks(x); ax.set_xticklabels(short, fontsize=9)
ax.set_ylabel("Fraction of Whisper Variance", fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_title("Sequential Variance Decomposition (adaptive k, threshold=0.95)\n"
             "(Acoustic ref: kaldi-librispeech  |  Textual ref: olmo-7b)", fontsize=11)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT + "/fig2_sequential_decomposition.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig2_sequential_decomposition.png")

# ── Figure 3: CKA double dissociation ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

subspaces = ["acoustic_unique_proj", "overlap_proj", "textual_unique_proj"]
sp_labels = ["Acoustic\nUnique", "Overlap\n(Joint)", "Textual\nUnique"]

for ax, ref, ref_label in zip(
    axes,
    ["kaldi", "llm"],
    ["Kaldi (acoustic reference)", "OLMo-7b (textual reference)"]
):
    enc_vals = {sp: [] for sp in subspaces}
    dec_vals = {sp: [] for sp in subspaces}

    for m in enc:
        mat = data["cka"][m]["matrix"]
        for sp in subspaces:
            v = mat.get(f"{sp}_vs_{ref}", mat.get(f"{ref}_vs_{sp}", float("nan")))
            enc_vals[sp].append(v)
    for m in dec:
        mat = data["cka"][m]["matrix"]
        for sp in subspaces:
            v = mat.get(f"{sp}_vs_{ref}", mat.get(f"{ref}_vs_{sp}", float("nan")))
            dec_vals[sp].append(v)

    x_sp = np.arange(len(subspaces))
    wb = 0.35
    enc_means = [np.nanmean(enc_vals[sp]) for sp in subspaces]
    dec_means = [np.nanmean(dec_vals[sp]) for sp in subspaces]
    enc_sds   = [np.nanstd(enc_vals[sp])  for sp in subspaces]
    dec_sds   = [np.nanstd(dec_vals[sp])  for sp in subspaces]

    ax.bar(x_sp - wb/2, enc_means, wb, yerr=enc_sds, capsize=4,
           color="#1565C0", label="Encoder", alpha=0.85)
    ax.bar(x_sp + wb/2, dec_means, wb, yerr=dec_sds, capsize=4,
           color="#42A5F5", label="Decoder", alpha=0.85)

    ax.set_xticks(x_sp); ax.set_xticklabels(sp_labels, fontsize=10)
    ax.set_ylabel("Linear CKA", fontsize=11)
    ax.set_ylim(0, 0.45)
    ax.set_title(f"CKA with {ref_label}", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle("CKA Double Dissociation: Unique Acoustic / Overlap / Unique Textual",
             fontsize=12)
plt.tight_layout()
plt.savefig(OUT + "/fig3_cka_dissociation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig3_cka_dissociation.png")

# ── Figure 4: Adaptive k values ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 4.5))
k_a = [data["results"][m]["k_acoustic_used"] for m in models]
k_t = [data["results"][m]["k_textual_used"]  for m in models]

ax.bar(x - 0.2, k_a, 0.4, color="#2196F3", label="k_acoustic", alpha=0.85)
ax.bar(x + 0.2, k_t, 0.4, color="#4CAF50", label="k_textual",  alpha=0.85)
ax.axhline(50, color="black", linestyle="--", lw=1.2, alpha=0.5, label="Fixed k=50 (previous)")
ax.axvline(len(enc)-0.5, color="black", linestyle="--", lw=1, alpha=0.4)
ax.text(len(enc)/2 - 0.5,          0.97, "Encoder", ha="center", fontsize=10,
        transform=ax.get_xaxis_transform())
ax.text(len(enc) + len(dec)/2 - 0.5, 0.97, "Decoder", ha="center", fontsize=10,
        transform=ax.get_xaxis_transform())
ax.set_xticks(x); ax.set_xticklabels(short, fontsize=9)
ax.set_ylabel("k (SVD directions retained)", fontsize=11)
ax.set_ylim(0, 55)
ax.set_title("Adaptive k Values per Model (threshold = 0.95 of cross-covariance SV^2 energy)", fontsize=11)
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT + "/fig4_adaptive_k.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig4_adaptive_k.png")

#!/usr/bin/env python3
"""Word-level GEOMETRY: mutual-kNN alignment matrix across models.

For each pair of representations, the mutual-kNN metric (Huh et al. 2024) measures
how many nearest neighbours each word shares between the two spaces — a local,
hubness-tolerant similarity that (unlike linear CKA) is invariant to the
anisotropic variance re-ranking that made CKA read near-zero here.

Reconstructed after the original scratchpad script was lost to a pod restart; the
result JSONs it produced are preserved in word/results/ (knn_pilot_results.json,
cknna_word_results.json). Word level so acoustic (mean-pooled frames) and LLM
(word tokens) are both 1:1 and honest.
"""
import json, time, pickle
import numpy as np
from pathlib import Path

WD = Path(__file__).resolve().parent.parent / "WordData"
MODELS = [
    "kaldi-librispeech", "wav2vec2-base", "parakeet-ctc-0.6b",   # acoustic
    "whisper-base-enc", "whisper-large-v2-enc",                  # whisper enc
    "whisper-base-dec", "whisper-large-v2-dec",                  # whisper dec
    "olmo-7b", "pythia-6.9b",                                    # LLM
]
K = 10; N_SUB = 15000; SEED = 42

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def load(name):
    with open(WD / f"word_embeddings_{name}.pkl", "rb") as f:
        return pickle.load(f)

def knn_sets(X):
    X = X.astype(np.float32); X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    idx = np.argpartition(-S, K, axis=1)[:, :K]
    return [set(r) for r in idx]

def main():
    rng = np.random.default_rng(SEED)
    N = len(load(MODELS[0]))
    sub = np.sort(rng.choice(N, size=min(N_SUB * 2, N), replace=False))
    feats = {}; valid = np.ones(len(sub), bool)
    for m in MODELS:
        e = load(m)[sub].astype(np.float32)
        valid &= np.linalg.norm(e, axis=1) > 1e-8
        feats[m] = e; log(f"loaded {m} {e.shape}")
    keep = np.where(valid)[0][:N_SUB]
    neigh = {m: knn_sets(feats[m][keep]) for m in MODELS}
    n = len(keep); M = len(MODELS)
    mat = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            ov = np.array([len(neigh[MODELS[i]][p] & neigh[MODELS[j]][p]) for p in range(n)])
            mat[i, j] = ov.mean() / K
    log(f"\n=== word-level mutual-kNN (k={K}, n={n}) ===")
    short = [s.replace("whisper-", "w-")[:12] for s in MODELS]
    print(" " * 13 + "".join(f"{s:>13s}" for s in short))
    for i, s in enumerate(short):
        print(f"{s:13s}" + "".join(f"{mat[i,j]:13.3f}" for j in range(M)))
    json.dump({"models": MODELS, "k": K, "n": int(n), "mutual_knn": mat.tolist()},
              open(Path(__file__).resolve().parent / "results" / "mutual_knn_matrix.json", "w"), indent=2)
    log("saved results/mutual_knn_matrix.json")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reconstruction-CKA with a shuffled-reference NULL baseline.

The paper's headline dissociation on one axis (CKA):
  raw            CKA(Whisper, single reference)                     -> LOW  (~0.11)
  reconstruction CKA(Whisper, held-out linear remix of references)  -> HIGH (~0.68)
  NULL           CKA(Whisper, remix of ROW-SHUFFLED references)      -> ~LOW (control)

The null breaks the row correspondence between Whisper and the references before
fitting, so any residual reconstruction-CKA is chance-level. A large gap between
reconstruction and null shows the 0.68 is genuine shared structure, not an
artifact of X̂ being a projection of X. Everything held-out (fit on train, CKA on
val). Phone level; references = kaldi + olmo + wav2vec2 + parakeet.
"""
import pickle, time, json
import numpy as np
from pathlib import Path

PD = Path("/dpluth-data/PhoneData")
OUT = Path(__file__).resolve().parent / "results"
REFS = ["kaldi-librispeech", "olmo-7b", "wav2vec2-base", "parakeet-ctc-0.6b"]
WHISPER = ["whisper-base-enc", "whisper-large-v2-enc", "whisper-base-dec", "whisper-large-v2-dec"]
N_SUB = 15000; SEED = 42; ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def load(n): return pickle.load(open(PD / f"phone_embeddings_{n}.pkl", "rb"))

def cka(Xv, Yv):
    Xc = Xv - Xv.mean(0); Yc = Yv - Yv.mean(0)
    num = (Xc.T @ Yc)
    den = np.linalg.norm(Xc.T @ Xc, "fro") * np.linalg.norm(Yc.T @ Yc, "fro")
    return float((num ** 2).sum() / den) if den > 0 else 0.0

def recon_cka(R, W, tr, va):
    """Fit Whisper(W) ~ R on train (alpha by val R2); return CKA(W_val, R_val@coef)."""
    Rc = R.astype(np.float64); Wc = W.astype(np.float64)
    mu = Rc[tr].mean(0); nu = Wc[tr].mean(0); Rc = Rc - mu; Wc = Wc - nu
    G = Rc[tr].T @ Rc[tr]; C = Rc[tr].T @ Wc[tr]; I = np.eye(G.shape[0])
    best = (-1e9, None)
    for a in ALPHAS:
        coef = np.linalg.solve(G + a * I, C); P = Rc[va] @ coef
        r2 = 1 - ((Wc[va] - P) ** 2).sum() / ((Wc[va] - Wc[va].mean(0)) ** 2).sum()
        if r2 > best[0]: best = (r2, coef)
    P = Rc[va] @ best[1]
    return cka(Wc[va].astype(np.float32), P.astype(np.float32)), float(best[0])

def main():
    rng = np.random.default_rng(SEED)
    N = len(load(REFS[0]))
    sub = np.sort(rng.choice(N, N_SUB, replace=False))
    perm = rng.permutation(N_SUB)                          # for the null (breaks correspondence)
    tr = rng.permutation(N_SUB)[:int(.7 * N_SUB)]
    va = np.setdiff1d(np.arange(N_SUB), tr)

    refs = []; valid = np.ones(N_SUB, bool)
    for r in REFS:
        e = load(r)[sub].astype(np.float32); valid &= np.linalg.norm(e, axis=1) > 1e-8
        refs.append(e - e.mean(0)); log(f"loaded {r} {e.shape}")
    R = np.concatenate(refs, axis=1)
    tr = tr[valid[tr]]; va = va[valid[va]]
    log(f"R dim={R.shape[1]}  train={len(tr)}  val={len(va)}")

    out = {}
    for wn in WHISPER:
        Xw = load(wn)[sub].astype(np.float32); Xw = Xw - Xw.mean(0)
        # raw single-reference CKA on val (geometry)
        raw = {r: cka(Xw[va], refs[i][va]) for i, r in enumerate(REFS)}
        rec, r2 = recon_cka(R, Xw, tr, va)                 # reconstruction-CKA + its R2
        null, _ = recon_cka(R[perm], Xw, tr, va)           # shuffled-reference null
        out[wn] = {"raw_cka": raw, "recon_cka": rec, "recon_r2": r2, "null_cka": null}
        log(f"{wn:22s} raw(olmo)={raw['olmo-7b']:.3f} raw(kaldi)={raw['kaldi-librispeech']:.3f}  "
            f"RECON={rec:.3f} (R2={r2:.3f})  NULL={null:.3f}")
        del Xw
    OUT.mkdir(exist_ok=True)
    json.dump({"refs": REFS, "n_val": int(len(va)), "results": out},
              open(OUT / "reconstruction_cka_results.json", "w"), indent=2)
    log("saved results/reconstruction_cka_results.json")

if __name__ == "__main__":
    main()

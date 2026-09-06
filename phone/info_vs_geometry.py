#!/usr/bin/env python3
"""Information vs Geometry at each Whisper layer (the paper's thesis, per layer).

For each layer and each anchor (kaldi=acoustic, olmo=word-identity/linguistic):
  GEOMETRY    = mutual-kNN alignment(layer, anchor)      [neighborhood similarity]
  INFORMATION = held-out ridge R2(anchor ~ layer)        [linear decodability]
The dissociation is INFORMATION >> GEOMETRY: the layer *contains* the target
structure (decodable) even where its *geometry* doesn't match it.
"""
import pickle, time, json
import numpy as np
from pathlib import Path
REPO = Path("/opt/modeling/zhoughton/misc/UGE-audio-text-models")
LED = REPO/"PhoneLayerData"/"layer_embeddings"; PD = Path("/dpluth-data/PhoneData")
MODELS = ["whisper-base", "whisper-large-v2"]
ANCHORS = ["kaldi-librispeech", "olmo-7b"]
K = 10; N = 12000; SEED = 42; ALPHAS = [1.0, 10.0, 100.0, 1000.0]

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def knn_sets(X, valid):
    X = X.astype(np.float32); X = X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-8)
    S = X@X.T; S[~valid,:]=-np.inf; S[:,~valid]=-np.inf; np.fill_diagonal(S,-np.inf)
    return [set(r) for r in np.argpartition(-S,K,axis=1)[:,:K]]

def align(a,b,valid): return float(np.mean([len(a[p]&b[p]) for p in np.where(valid)[0]])/K)

def ridge_r2(X, Y, tr, va):
    Xc = X.astype(np.float64); Yc = Y.astype(np.float64)
    mu = Xc[tr].mean(0); nu = Yc[tr].mean(0)
    Xc = Xc-mu; Yc = Yc-nu
    G = Xc[tr].T@Xc[tr]; C = Xc[tr].T@Yc[tr]; I = np.eye(G.shape[0]); best=-1e9
    for a in ALPHAS:
        W = np.linalg.solve(G+a*I, C); P = Xc[va]@W
        ss_res = ((Yc[va]-P)**2).sum(); ss_tot = ((Yc[va]-Yc[va].mean(0))**2).sum()
        best = max(best, 1-ss_res/ss_tot)
    return float(best)

def main():
    sub = np.load(REPO/"PhoneLayerData"/"phone_sub_indices.npy")
    rng = np.random.default_rng(SEED)
    pos = np.sort(rng.choice(len(sub), N, replace=False)); orig = sub[pos]
    perm = rng.permutation(N); tr = perm[:int(.7*N)]; va = perm[int(.7*N):]

    anc = {}; aknn = {}; valid = np.ones(N, bool)
    for a in ANCHORS:
        E = pickle.load(open(PD/f"phone_embeddings_{a}.pkl","rb")); A = E[orig].astype(np.float32); del E
        valid &= np.linalg.norm(A,axis=1)>1e-8; anc[a]=A
    for a in ANCHORS: aknn[a]=knn_sets(anc[a], valid)
    log(f"anchors ready; valid={int(valid.sum())}")

    out = {}
    for M in MODELS:
        out[M] = {}
        for comp in ["enc","dec"]:
            files = sorted(LED.glob(f"phone_layer_emb_{M}-{comp}_layer*.pkl")); nL=len(files)
            log(f"\n{M}-{comp}: {nL} layers")
            print(f"  {'lyr':>3s} {'dep':>4s} | {'geo_kald':>8s} {'inf_kald':>8s} {'gap_k':>6s} | "
                  f"{'geo_olmo':>8s} {'inf_olmo':>8s} {'gap_o':>6s}")
            rows=[]
            for f in files:
                L=int(f.stem.split("layer")[-1]); X=pickle.load(open(f,"rb"))[pos]
                lk=knn_sets(X, valid); d=L/(nL-1) if nL>1 else 0.0
                gk=align(lk,aknn["kaldi-librispeech"],valid); ik=ridge_r2(X,anc["kaldi-librispeech"],tr,va)
                go=align(lk,aknn["olmo-7b"],valid);            io=ridge_r2(X,anc["olmo-7b"],tr,va)
                print(f"  {L:3d} {d:4.2f} | {gk:8.3f} {ik:8.3f} {ik-gk:+6.3f} | {go:8.3f} {io:8.3f} {io-go:+6.3f}")
                rows.append({"layer":L,"depth":d,"geo_kaldi":gk,"inf_kaldi":ik,
                             "geo_olmo":go,"inf_olmo":io})
                del X,lk
            out[M][comp]=rows
    json.dump({"k":K,"n":N,"anchors":ANCHORS,"results":out},
              open(Path(__file__).resolve().parent/"results"/"info_vs_geom_results.json","w"), indent=2)
    log("DONE")

if __name__=="__main__": main()

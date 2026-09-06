#!/usr/bin/env python3
"""Layer trajectory across ALL extracted Whisper sizes (base/small/medium/large-v2).
Same anchors + subsample as the base pilot; adds normalized depth for cross-size
comparison. Confirms whether the acoustic->abstract encoder sweep + linguistic
decoder is general and how it scales."""
import pickle, time, json
import numpy as np
from pathlib import Path
REPO = Path("/opt/modeling/zhoughton/misc/UGE-audio-text-models")
LED = REPO/"PhoneLayerData"/"layer_embeddings"; PD = Path("/dpluth-data/PhoneData")
MODELS=["whisper-base","whisper-small","whisper-medium","whisper-large-v2"]
K=10; N_SUB=12000; SEED=42
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)
def knn_sets(X,valid):
    X=X.astype(np.float32); X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-8)
    S=X@X.T; S[~valid,:]=-np.inf; S[:,~valid]=-np.inf; np.fill_diagonal(S,-np.inf)
    return [set(r) for r in np.argpartition(-S,K,axis=1)[:,:K]]
def align(a,b,valid): return float(np.mean([len(a[p]&b[p]) for p in np.where(valid)[0]])/K)
def pr(X,valid):
    X=X[valid].astype(np.float64); X-=X.mean(0); v=(X**2).sum(0)
    return float(v.sum()**2/(v**2).sum())
def main():
    sub=np.load(REPO/"PhoneLayerData"/"phone_sub_indices.npy")
    rng=np.random.default_rng(SEED); pos=np.sort(rng.choice(len(sub),N_SUB,replace=False)); orig=sub[pos]
    log("anchors...")
    anc={}; valid=np.ones(N_SUB,bool)
    for n in ["kaldi-librispeech","wav2vec2-base","olmo-7b"]:
        E=pickle.load(open(PD/f"phone_embeddings_{n}.pkl","rb")); A=E[orig].astype(np.float32); del E
        valid&=np.linalg.norm(A,axis=1)>1e-8; anc[n]=A
    aknn={n:knn_sets(anc[n],valid) for n in anc}
    out={}
    for M in MODELS:
        out[M]={}
        for comp in ["enc","dec"]:
            files=sorted(LED.glob(f"phone_layer_emb_{M}-{comp}_layer*.pkl"))
            nL=len(files); rows=[]
            log(f"{M}-{comp}: {nL} layers")
            print(f"  {'layer':>5s} {'depth':>5s} {'kaldi':>7s} {'wav2vec2':>9s} {'olmo(wID)':>10s} {'pRatio':>8s}")
            for f in files:
                L=int(f.stem.split("layer")[-1]); X=pickle.load(open(f,"rb"))[pos]
                lk=knn_sets(X,valid)
                ak=align(lk,aknn["kaldi-librispeech"],valid); aw=align(lk,aknn["wav2vec2-base"],valid)
                ao=align(lk,aknn["olmo-7b"],valid); p=pr(X,valid); d=L/(nL-1) if nL>1 else 0.0
                print(f"  {L:5d} {d:5.2f} {ak:7.3f} {aw:9.3f} {ao:10.3f} {p:8.1f}")
                rows.append({"layer":L,"depth":d,"kaldi":ak,"wav2vec2":aw,"olmo_wordID":ao,"part_ratio":p})
                del X,lk
            out[M][comp]=rows
    json.dump({"k":K,"n":N_SUB,"results":out},open(Path(__file__).resolve().parent/"results"/"layer_traj_all_results.json","w"),indent=2)
    log("DONE")
if __name__=="__main__": main()

# Speech, Acoustic, and Language Model Representations: Information vs Geometry

Do speech models (Whisper), acoustic models (Kaldi, wav2vec2, parakeet), and
language models (OLMo, Pythia, …) represent the same information? Do they
represent it the **same way**?

**Headline finding: they share information but not geometry.** Across every
*geometric* similarity metric (linear CKA, mutual-kNN, principal-subspace overlap),
Whisper's representations are distinct from acoustic and language models — and
acoustic models are distinct from each other; only language models converge with
one another. Yet *information* metrics (held-out ridge predictivity, and CKA
between a representation and its reconstruction from the others) show the
underlying information is **substantially shared**. The geometry differs because
each training objective imposes its own similarity structure and re-bases/re-ranks
shared information onto different axes — not because information is lost.

The mechanism is visible **per layer**: Whisper's encoder *retains* acoustic
information (ridge R² ≈ 0.38 even at the output) inside a geometry that no longer
looks acoustic (mutual-kNN ≈ 0.07–0.14) — an information−geometry gap of +0.2–0.3
at every layer, robust across model scale. The decoder instead *discards* acoustic
information and carries linguistic information in a geometry that more faithfully
tracks it. See the money plot: `phone/figures/info_vs_geometry_by_layer.png`.

> Note on metrics: the geometric verdict rests on **CKA + mutual-kNN + subspace
> overlap** (which agree). An earlier CKNNA implementation was buggy and is not
> used — rely on mutual-kNN (as in Huh et al. 2024) if a hubness-robust metric is
> wanted.

---

## Repository structure

See **`STRUCTURE.md`** for the full layout. In short:

```
extraction/   Shared embedding-extraction engine (run from inside this dir)
phone/        Phone-level analyses  — layer_trajectory.py, info_vs_geometry.py, plot_*, results/, figures/
word/         Word-level analyses   — geometry_mutual_knn.py, results/, figures/
archived/     Superseded / other-corpus scripts, plots, and run logs
```

Granularity matters: acoustic comparisons run at **phone** level (within-word
detail is real); LLM comparisons run at **word** level (phone-level LLM embeddings
are word-broadcast, hence degenerate). Large embedding/audio caches are gitignored.

---

## Models

### Audio models (audio input)

| Name | HuggingFace ID | Params | Architecture |
|---|---|---|---|
| whisper-{base,small,medium,large}-{enc,dec} | openai/whisper-* | 74M–1.55B | Whisper encoder / decoder |
| parakeet-ctc-0.6b | nvidia/parakeet-ctc-0.6b | 600M | FastConformer-CTC |
| wav2vec2-base | facebook/wav2vec2-base | 95M | Self-supervised (contrastive) |
| mimi | kyutai/mimi | ~85M | Conv+Transformer neural codec |
| kaldi-librispeech | (local chain model) | — | Hybrid HMM-DNN acoustic (senone bottleneck) |

### Text LLMs (text input, text training)

| Name | HuggingFace ID | Params | Corpus |
|---|---|---|---|
| babylm-{125m,350m,1.3b} | znhoughton/opt-babylm-* | 125M–1.3B | BabyLM (~100M tokens) |
| opt-125m | facebook/opt-125m | 125M | ~180B tokens |
| pythia-{160m,6.9b} | EleutherAI/pythia-* | 160M–6.9B | The Pile |
| olmo-7b | allenai/OLMo-2-1124-7B | 7B | Dolma |

**Key controls:** babylm-125m vs opt-125m (data volume); pythia-160m vs pythia-6.9b
(model size); OLMo-7B vs Pythia-6.9B (architecture/corpus); Whisper base→large
(audio-encoder scaling); Whisper enc vs dec (network component).

Corpus: **LJSpeech** (single speaker), force-aligned to phones/words.

---

## Setup

```bash
pip install -r requirements.txt
pip install --upgrade transformers accelerate   # OLMo-2 needs transformers>=4.48
huggingface-cli login                            # then accept the OLMo-2 license page
export HF_HOME=/your/large/volume/huggingface_cache
```

On the internal pod, use the working interpreter
`/opt/modeling/zhoughton/envs/propsensity_eval/bin/python` (pyenv 3.12.4; the
default pyenv shim is glibc-broken). **GPU is only visible outside the Claude Code
sandbox** — launch extraction jobs from a normal pod shell. The pod has a ~150 GB
cgroup limit with group-OOM, so keep peak RAM bounded (the per-layer extractor is
float16 + a 150k utterance subsample for exactly this reason).

---

## Reproducing the analysis

**1. Extract embeddings** (GPU, from `extraction/`):
```bash
cd extraction
python phone_level_analysis.py   # phone embeddings → PhoneData/
python word_level_analysis.py    # word embeddings  → WordData/
python whisper_layer_similarity.py --root_dir .. --dataset phone \
       --models whisper-base whisper-small whisper-medium whisper-large-v2
       # per-layer phone embeddings → PhoneLayerData/layer_embeddings/
```

**2. Information (variance decomposition / predictivity):**
```bash
bash extraction/run_projection_analysis.sh      # writes phone/results, word/results
```

**3. Geometry + the capstone** (CPU):
```bash
python phone/layer_trajectory.py        # per-layer mutual-kNN to acoustic / word-identity anchors
python phone/info_vs_geometry.py        # information vs geometry per layer  → phone/results/
python phone/plot_info_vs_geometry.py   # → phone/figures/info_vs_geometry_by_layer.png
python word/geometry_mutual_knn.py      # word-level mutual-kNN matrix       → word/results/
```

---

## What to look for

- **Geometry (mutual-kNN / CKA):** cross-family alignment is low; within-family
  (LLM↔LLM, encoder↔encoder) is high; no coherent "acoustic pole."
- **Information (ridge R²):** high where geometry is low — the dissociation.
- **Layer trajectory:** encoder acoustic-alignment peaks at ~70% depth then
  abstracts away; decoder is linguistic throughout; final layers collapse in
  effective dimensionality.

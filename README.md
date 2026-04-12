# Representation Analysis: Cross-Modal CKA & PCA

Tests the **Platonic Representation Hypothesis** by comparing internal representations
of Whisper (audio) and three LLMs (text) on paired LibriSpeech utterances.

## Setup

```bash
pip install -r requirements.txt
```

For OLMo-2 you may need the latest transformers from source:
```bash
pip install --upgrade git+https://github.com/huggingface/transformers.git
```

## Usage

```bash
# Default: 200 samples, results saved to ./results/
python representation_analysis.py

# Larger run with GPU
python representation_analysis.py --n_samples 500 --output_dir ./results_500

# Re-use cached embeddings (re-run analysis without re-extracting)
python representation_analysis.py --skip_existing
```

## What it does

### Step 1 — Data
Downloads `openslr/librispeech_asr` (test-clean split) from HuggingFace.
Each sample has paired audio + transcript.

### Step 2 — Embeddings
Models are loaded one at a time to keep VRAM usage manageable.

| Model | Modality | Pooling |
|---|---|---|
| openai/whisper-base | Audio | Mean over encoder time frames |
| znhoughton/opt-babylm-125m | Text | Mean over non-padding tokens (last hidden state) |
| znhoughton/opt-babylm-1.3b | Text | Mean over non-padding tokens (last hidden state) |
| allenai/OLMo-2-1124-7B | Text | Mean over non-padding tokens (last hidden state) |
| EleutherAI/pythia-6.9b | Text | Mean over non-padding tokens (last hidden state) |

OLMo-2 and Pythia-6.9b are matched on scale (~7B params) but differ in architecture
(OLMo-2: standard MHA + non-parametric LayerNorm, Dolma corpus; Pythia: GPT-NeoX with
parallel attention+MLP layers, trained on The Pile). This makes their pairwise CKA the
key "architecture vs. scale" control — same scale, different design and data lineage.

Embeddings are cached as `.pkl` files so you don't need to re-run expensive models.

### Step 3 — PCA Eigenvalue Spectra (`eigenspectra.png`)
Shows the variance explained by each principal component (log scale).
Similar decay curves across models → similar intrinsic dimensionality → similar
geometric structure.

### Step 4 — Pairwise CKA (`cka_heatmap.png`)
Linear CKA between every pair of model representations.
- **1.0** = identical geometry
- **0.0** = no shared structure
- Cross-modal score (Whisper vs LLM) > ~0.3 is notable evidence for shared representations.

### Step 5 — 2D PCA Scatter (`pca_scatter.png`)
Visual check of clustering structure in each model's space.

## Output files

```
results/
├── embeddings_whisper-base.pkl       # cached (N, hidden_dim) arrays
├── embeddings_babylm-125m.pkl
├── embeddings_babylm-1.3b.pkl
├── embeddings_olmo-7b.pkl
├── eigenspectra.png                   # PCA decay curves
├── cka_heatmap.png                    # pairwise CKA matrix
├── pca_scatter.png                    # 2D projections
└── results.json                       # all scores as JSON
```

## What to look for

**Within-modality CKA (LLM vs LLM):**
- BabyLM-125m vs BabyLM-1.3b: likely high (same architecture family, similar data)
- BabyLM vs OLMo/Mistral: lower — different training data and scale
- **OLMo-7B vs Pythia-6.9b**: the key comparison — same scale, different architecture
  (GPT-NeoX vs OLMo-2) and different training data (The Pile vs Dolma). High CKA here =
  convergence is scale-driven, not architecture- or data-specific.
- If both 7B models are *more* similar to each other than either is to BabyLM, that
  supports the Platonic Representation Hypothesis (scale → convergence).

**Cross-modal CKA (Whisper vs any LLM):**
- Near-zero: audio and text learn incompatible representations
- Notably positive: supports convergence toward a shared statistical model of reality

## Notes on pooling choice

Mean-pooling the last hidden layer is a reasonable default, but it's worth knowing:
- For LLMs, earlier layers sometimes capture syntax better; later layers capture semantics
- You could modify `extract_lm_embeddings` to extract a specific layer index via
  `out.hidden_states[layer_idx]` instead of `[-1]`
- For Whisper, the encoder output already encodes the full utterance context

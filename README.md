# Representation Analysis: Cross-Modal Minibatch CKA & PCA

Tests the **Platonic Representation Hypothesis** — the conjecture that neural networks
trained on different data and modalities converge toward a shared statistical model of
reality — by comparing internal representations of audio and text models on paired
LibriSpeech utterances.

## Scientific Background

The core question: if you run the same utterance through an audio model (which only hears
it) and a language model (which only reads its transcript), do the resulting embedding
vectors have similar geometric structure? If yes, that's evidence both models are learning
the same underlying representation of meaning despite never sharing data or training
objectives.

**Key papers:**
- Huh et al. (2024) "The Platonic Representation Hypothesis" — ICML 2024
- Kornblith et al. (2019) "Similarity of Neural Network Representations Revisited" — original CKA paper
- Nguyen, Raghu & Kornblith (2021) "Do Wide and Deep Networks Learn the Same Things?" — minibatch CKA
- Murphy, Zylberberg & Fyshe (2024) "Correcting Biased CKA Measures" — `arxiv:2405.01012`
- Horoi et al. "Deceiving the CKA Similarity Measure" — outlier sensitivity

---

## Models

| Name | HuggingFace ID | Modality | Params | Architecture | Training Data |
|---|---|---|---|---|---|
| whisper-base | openai/whisper-base | Audio | 74M | Transformer encoder | 680k hrs multilingual |
| parakeet-ctc-0.6b | nvidia/parakeet-ctc-0.6b | Audio | 600M | FastConformer-CTC | Granary 64k hrs English |
| babylm-125m | znhoughton/opt-babylm-125m-20eps-seed964 | Text | 125M | OPT | BabyLM corpus |
| babylm-1.3b | znhoughton/opt-babylm-1.3b-20eps-seed964 | Text | 1.3B | OPT | BabyLM corpus |
| olmo-7b | allenai/OLMo-2-1124-7B | Text | 7B | OLMo-2 | Dolma |
| pythia-6.9b | EleutherAI/pythia-6.9b | Text | 6.9B | GPT-NeoX | The Pile |

**OLMo-7B vs Pythia-6.9B** is the key architectural control — matched scale (~7B),
different architecture (OLMo-2 vs GPT-NeoX) and different training data (Dolma vs The
Pile). High CKA between them means convergence is scale-driven, not architecture-specific.

**Whisper vs Parakeet** is the audio architecture control — both audio encoders,
different architecture (Transformer vs FastConformer) and different training data.

---

## Setup

```bash
pip install -r requirements.txt
pip install --upgrade accelerate transformers
```

OLMo-2 requires transformers >= 4.48 and accelerate >= 1.0:
```bash
pip install --upgrade transformers accelerate
```

**HuggingFace authentication** — OLMo-2 requires accepting a license:
```bash
huggingface-cli login
# Then visit https://huggingface.co/allenai/OLMo-2-1124-7B and accept the license
```

**Environment variables** — point HuggingFace cache to your large storage volume:
```bash
export HF_HOME=/workspace/huggingface_cache
export HF_DATASETS_CACHE=/workspace/huggingface_cache/datasets
```

Add these to `~/.bashrc` to persist across sessions.

---

## Usage

```bash
# Full run — all LibriSpeech splits (~292k utterances)
python representation_analysis.py

# Quick test run
python representation_analysis.py --splits test --max_samples 500

# Restart after interruption — automatically resumes from checkpoints and cached embeddings
python representation_analysis.py

# Skip stability check for faster iteration
python representation_analysis.py --skip_stability

# Tune batch sizes for your GPU setup
python representation_analysis.py \
    --whisper_batch_size 1024 \
    --parakeet_batch_size 128 \
    --lm_batch_size 64
```

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--splits` | all 7 splits | LibriSpeech splits to use |
| `--max_samples` | None | Cap total samples (useful for testing) |
| `--root_dir` | `.` | Root directory for Data/, Plots/, logs/ |
| `--skip_extraction` | False | Skip extraction even if no cache exists |
| `--batch_size` | 2048 | Minibatch size for CKA computation |
| `--whisper_batch_size` | 1024 | Batch size for Whisper audio extraction |
| `--parakeet_batch_size` | 128 | Batch size for Parakeet audio extraction |
| `--lm_batch_size` | 64 | Batch size for LM forward passes |
| `--pca_components` | 50 | Number of PCA components for eigenvalue analysis |
| `--skip_stability` | False | Skip outlier stability check |

---

## How It Works

### Step 1 — Data
Downloads all 7 LibriSpeech splits (~292k utterances total) from HuggingFace. Each
sample has paired audio + text transcript. Audio goes to the audio models, transcripts
go to the LLMs — same underlying utterance, two modalities.

Transcripts are cached to `Data/texts_*.json` after the first run, saving ~15 minutes
on every subsequent restart.

### Step 2 — Embedding Extraction
Models are loaded one at a time (one model in VRAM at a time) using `device_map="auto"`
to spread large models across multiple GPUs. Each model produces one embedding vector
per utterance via **mean pooling** over its final hidden layer:

- **Audio models**: mean pool over encoder time frames
- **Text models**: mean pool over non-padding token positions

Embeddings are cached as `.pkl` files in `Data/`. Extraction automatically resumes
from mid-run checkpoints (saved every 500 batches) if interrupted — just restart the
script with no flags.

### Step 3 — PCA Eigenvalue Analysis
For each model's embedding matrix, computes the top-50 eigenvalues of the covariance
matrix via SVD. Eigenvalues are normalised to sum to 1 (fraction of variance explained).
Similar decay curves across models indicate similar intrinsic dimensionality.

**Effective rank** = exp(Shannon entropy of eigenvalue distribution) — a single number
summarising how many dimensions the model is actually using.

### Step 4 — Pairwise Minibatch CKA
Computes linear CKA between every pair of models using the **debiased** HSIC estimator
(Szekely & Rizzo 2014 kernel centering, as recommended by Kornblith et al. 2019 and
Murphy et al. 2024). The biased estimator inflates similarity scores when feature
dimensionality exceeds sample count — using the debiased version avoids this. Minibatch
accumulation follows Nguyen et al. (2021): data is processed in non-overlapping
minibatches of 2048 samples, averaging HSIC scores across batches before the final
normalisation step. This avoids the O(N²) memory cost of full CKA while using all 292k
samples.

CKA is invariant to rotation and isotropic scaling — it measures genuine geometric
similarity, not coordinate accidents.

### Step 5 — Outlier Stability Check
Runs CKA 10 times on random 80% subsets per model pair. High variance (std > 0.03)
flags potential outlier sensitivity per Horoi et al.

---

## Output Structure

```
Data/
├── texts_*.json                    # cached transcripts (fast restart)
├── embeddings_whisper-base.pkl     # (N, 512) float32
├── embeddings_parakeet-ctc-0.6b.pkl # (N, 1024) float32
├── embeddings_babylm-125m.pkl      # (N, 768) float32
├── embeddings_babylm-1.3b.pkl      # (N, 2048) float32
├── embeddings_olmo-7b.pkl          # (N, 4096) float32
├── embeddings_pythia-6.9b.pkl      # (N, 4096) float32
├── *_checkpoint.pkl                # mid-run checkpoints (auto-deleted on completion)
├── cka_matrix.csv                  # full N×N pairwise CKA scores
├── cka_cross_modal.csv             # audio vs LLM scores with metadata
├── pca_summary.csv                 # effective rank, variance per component
├── cka_stability.csv               # mean, std, min, max per pair
└── results.json                    # all results in one file

Plots/
├── eigenspectra.png                # per-model PCA decay curves
├── eigenspectra_overlay.png        # all models on one axes
├── effective_rank.png              # bar chart of effective rank per model
├── pca_scatter.png                 # 2D PCA projections
├── cka_heatmap.png                 # pairwise CKA matrix heatmap
├── cka_cross_modal_bar.png         # grouped bar: each audio model vs each LLM
└── cka_stability.png               # box plots of CKA variance across subsets

logs/
└── run_YYYYMMDD_HHMMSS.log        # full timestamped run log with GPU memory stats
```

---

## What to Look For

**Cross-modal CKA (audio ↔ LLM):**
- CKA > ~0.3 between Whisper/Parakeet and any LLM is notable evidence for the
  Platonic Representation Hypothesis
- If larger models (OLMo, Pythia) are more similar to audio models than BabyLM,
  that supports the idea that scale drives convergence toward universal representations

**Within-modality text CKA:**
- BabyLM-125M vs BabyLM-1.3B: expected to be high (same architecture and data)
- BabyLM vs 7B models: expected to be lower (different scale and data)
- OLMo-7B vs Pythia-6.9B: the key control — high CKA = scale drives convergence,
  low CKA = architecture/data choices matter more than scale

**Audio-audio CKA (Whisper vs Parakeet):**
- Both are audio encoders but with different architectures and training data
- High CKA here would suggest audio representations converge regardless of architecture
- Compare to text-text pairs of similar size difference for context

**Eigenvalue spectra:**
- Steep decay = model compresses information into few dimensions
- Flat decay = model uses its full dimensional capacity
- Similar decay curves across audio and text models = similar intrinsic geometry

**Effective rank:**
- Whisper-base (512 hidden dim) vs Parakeet (1024 hidden dim) vs LLMs (768–4096)
- Effective rank normalises for hidden dim size — it's about how many dimensions
  are actually being used, not how many exist

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 16 GB | 2× A100 80GB (160 GB total) |
| RAM | 64 GB | 200+ GB (for full dataset loading) |
| Disk | 200 GB | 450 GB (dataset ~300 GB + models ~33 GB + embeddings ~20 GB) |

With 2× A100 80GB, use `max_memory={0: "55GiB", 1: "75GiB"}` in model loading if
another process is sharing GPU 0 (~17 GB). This is already set in the script.

---

## Testing

Before running the full script, validate the data pipeline and model I/O:

```bash
python test_dataset.py
```

Tests: dataset loading, `select()` vs `iter()` on concatenated splits, Whisper processor
output shape, Parakeet feature extractor key names and forward pass, LM hidden states
and mean pooling. All 9 sections should pass before committing to a multi-hour run.

---

## Restarts and Fault Tolerance

The script is designed to resume gracefully:

- **Completed models**: embeddings cached in `Data/embeddings_*.pkl`, loaded automatically on restart
- **Partially completed models**: checkpoint saved every 500 batches to `Data/*_checkpoint.pkl`, resumed automatically
- **Transcripts**: cached to `Data/texts_*.json` after first extraction, skips 15-minute wait on restart
- **No flags needed**: just run `python representation_analysis.py` again — it figures out what's done and what isn't

---

## .gitignore

```
Data/embeddings_*.pkl
Data/*_checkpoint.pkl
Data/texts_*.json
logs/
__pycache__/
*.pyc
.env
huggingface_cache/
.ipynb_checkpoints/
```

The results CSVs and `results.json` in `Data/` are small and worth committing.
The `.pkl` files are large (up to 2.3 GB each) and fully reproducible — do not commit them.

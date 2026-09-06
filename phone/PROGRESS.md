# Whisper–Kaldi–LLM Variance Decomposition — Progress

_Last updated: 2026-08-14_

**Status:** grouped split, collateral diagnostic, and the separability hierarchy
(replacing the old MLP verification) are pushed to `main`; leak-free re-run of
both levels pending on the server. Hierarchy validated on synthetic data
(separable target → interaction gap ≈ 0; joint target → gap ≈ +0.22).

## Research question / claim

Does the Platonic Representation Hypothesis (PRH) hold for Whisper? Our claim:
**PRH does not hold, because including multiple modalities is not analogous to
simply combining features.** Whisper's embeddings are not a simple combination
of acoustic (Kaldi) and linguistic (LLM) features.

We are building this argument in layers:

1. **Linear level (done):** Whisper is not a simple *linear* combination
   `Lₐ(a) + L_b(b)` of acoustic + language features — a large fraction of
   Whisper variance is in "neither" linear subspace.
2. **Nonlinear level (next):** rule out a *separable nonlinear* combination
   `f(a) + g(b)`, while explicitly **not** ruling out a *nonseparable joint*
   function `F(a, b)`. A nonseparable representation (information fused, not
   added) is consistent with our anti-PRH argument.

- `a` = Kaldi acoustic embedding, `b` = OLMo (`olmo-7b`) LLM embedding, both
  used as proxies for the acoustic/linguistic modalities.

## Method (current)

**Forward ridge regression, commonality analysis, held-out, grouped split.**

For each Whisper model, regress Whisper **on** the target (target = predictor),
subtract the projection, and partition Whisper's variance:

- `linear_deflation`: fit `Whisper ≈ X_target · W` (ridge, α by held-out CV) on
  train, subtract → residual = the part of Whisper not linearly predictable
  from the target. Whisper is the response so its total variance is the
  denominator for every fraction.
- `venn_decomposition`: run both orderings (Kaldi-first, LLM-first) to recover
  four components via commonality analysis:
  - `kaldi_only` — acoustic, not language
  - `llm_only` — language, not acoustic
  - `shared` — in both linear subspaces
  - `neither` — in neither linear subspace
- All variance fractions measured on the **val split only** (held-out).
- **Nonlinear verification (being redesigned — see Open issues):** MLP probes
  on the "neither" residual.

Memory-safe for OLMo (4096-dim, ~14.5 GB f16): chunked Gram/cross accumulation
(never a full float32 copy), float16 residuals.

## Key methodological decisions (and why)

- **Forward direction (regress Whisper *on* target), not reverse.** To decompose
  *Whisper's* variance, Whisper must be the response. The reverse (regress
  target on Whisper, INLP-style nullspace projection) is ill-posed when the
  target has more dims than Whisper (OLMo 4096 > Whisper ≤1280): the coefficient
  span becomes all of Whisper and the method trivially "explains" 100%.
- **No PCA rank-matching.** We tried PCA-reducing both targets to a common rank
  to equalize the Kaldi(512)-vs-LLM(4096) budget, but truncating by each
  target's *own* variance is circular w.r.t. the question and hard to defend.
- **Held-out measurement handles the dimensionality confound instead.** The
  concern that a linear fit "subtracts noise" from non-target Whisper dims is
  bounded by `rank(target)/n` (finite-sample overlap), ≈0.07% at our n, and →0
  out-of-sample. Verified in simulation (in-sample vs held-out gap = the wrongly
  subtracted fraction; identical at d_whisper=100 and 1280). n/p ≈ 171 (OLMo)
  to 1370 (Kaldi) — plenty of samples.
- **Grouped train/val split (`--group_by`, default `utt_id`).** A random split
  let repeated/near-identical rows (same utterance) land in both splits, which
  a high-capacity MLP exploited (see below). Grouping by an item id keeps every
  item entirely in train or val.

## Results so far (⚠️ from the LEAKY random-split run — re-running with grouped split)

Linear decomposition is self-consistent (V_kaldi_total matches ord1 R², etc.)
and the pattern is sensible:

| level / modality | kaldi_only | llm_only | shared | neither | V_kaldi | V_llm |
|---|---|---|---|---|---|---|
| word encoders  | 0.02–0.04 | 0.29–0.36 | 0.15–0.20 | ~0.45 | 0.17–0.23 | 0.50–0.56 |
| word decoders  | ~0.00–0.01 | 0.16–0.32 | 0.05–0.12 | 0.54–0.77 | 0.06–0.12 | 0.22–0.47 |
| phone encoders | 0.07–0.10 | 0.20–0.29 | 0.06–0.08 | ~0.59 | 0.13–0.17 | 0.28–0.36 |
| phone decoders | ~0.00 | 0.22–0.40 | 0.02–0.06 | 0.53–0.75 | 0.03–0.06 | 0.25–0.45 |

- **Encoders are more acoustic** than decoders (higher `kaldi_only`, `V_kaldi`);
  **phone encoders more than word encoders** — as expected.
- **Decoders align more with the LLM, barely with Kaldi** (`kaldi_only ≈ 0`) —
  a PRH-relevant story: Whisper's decoder behaves more language-model-like.
- **`neither` is the largest bucket everywhere** (~0.45–0.77) — the core support
  for "not a simple linear combination."
- Caveat: `llm_only ≫ kaldi_only` partly reflects OLMo being 4096-dim vs Kaldi
  ~512 (no rank-matching); the *qualitative* encoder-vs-decoder contrast is
  robust, but hedge the raw Kaldi-vs-LLM magnitude comparison.
- Minor: `shared` vs `shared_check` and the two `neither` orderings differ by
  ~0.025 (just over the 0.02 warn threshold) — mild Kaldi/LLM collinearity.

## Open issues

### 1. MLP verification leakage → `nlr2 = 1.0` everywhere (FIX PUSHED, re-running)
The nonlinear MLP hit **held-out R² = exactly 1.0 for all 24 model×target runs**.
Real training converging to a flat 1.0000 is train/val **leakage from repeated
rows** — the MLP memorized a lookup table. Fixed with the grouped split.
- ⚠️ `utt_id` grouping may not fully fix it if Whisper embeddings are
  word/phone-**type**-determined (same type in both splits). If R² stays ~1.0,
  escalate to `--group_by word` / `phone` (hold out types — the stricter test).
- Linear numbers are robust to this (ridge can't memorize), but will be
  re-confirmed leak-free.

### 2. MLP verification conflates `f(a)+g(b)` and `F(a,b)` — IMPLEMENTED
A single MLP is a universal approximator: it captures separable and nonseparable
structure indistinguishably, so it could not support the core claim. Replaced
with a model hierarchy (`separability_analysis`), scored by grouped held-out R².
Output columns: `r2_linear_additive`, `r2_nonlinear_separable`,
`r2_nonlinear_joint`, `gap_nonlinear_separable`, `gap_interaction` (replacing
the old `nlr2_kaldi` / `nlr2_llm`).

| model | form | captures |
|---|---|---|
| 1. linear-additive | `W ≈ Lₐ(a) + L_b(b)` | linear separable (current decomposition) |
| 2. nonlinear-separable | `W ≈ f(a) + g(b)` (two towers, summed) | + nonlinear separable |
| 3. nonlinear-joint | `W ≈ F([a;b])` (concat input) | + nonseparable interaction |

- **(2) − (1)** = nonlinear-but-separable structure — the `f(a)+g(b)` to rule out.
- **(3) − (2)** = **nonseparable interaction** — the `F(a,b)` to keep. **The gap
  is the effect size.** Claim supported iff **(3) ≫ (2)**.
- This is a functional-ANOVA / GAM-vs-full-model interaction test (the gap ≈ a
  Sobol interaction index). Kaldi/LLM correlation *shrinks* the gap, so a
  positive gap is conservative evidence.

Architecture / implementation notes:
- **Model 2 (separable):** two independent MLP towers `f(a)`, `g(b)`, each
  producing `d_whisper`, summed at the output; jointly trained (MSE to `W`).
  The additivity is the only constraint — each tower gets full depth/width.
- **Model 3 (joint):** one MLP on `concat([a, b])`. **Capacity-match ≤ the two
  towers combined** (params), so a positive `(3)−(2)` gap can't be "more
  parameters," only nonseparability.
- Reuse the existing `train_probe` machinery (GPU-resident, bf16 AMP, early
  stop) and the grouped `val_mask`. Score pooled held-out R² on `W` (the full
  Whisper embedding), not on the linear "neither" residual, so the three-model
  hierarchy is directly comparable.
- Caveat to state: `a`, `b` are Kaldi/OLMo embeddings (proxies for the
  modalities), so the claim is "Whisper is a nonseparable function of these two
  representations."

### 3. Per-fit collateral diagnostic — DONE (`6a758b1`)
`linear_deflation` now also returns in-sample R² (reuses the fitted coefficients,
no re-fit). Logged per fit and stored in the JSON as `insample_r2` /
`collateral_gap` (= in-sample − held-out). Expect `≈ rank(target)/n`
(~0.07% Kaldi, up to ~0.6% OLMo); a materially larger gap flags a problem.

## Files

- `whisper_kaldi_llm_projection/projection_analysis.py` — main analysis.
- `run_projection_analysis.sh` — runs word then phone, each sequential, all 12
  Whisper models.
- Outputs: `whisper_kaldi_llm_projection/{word,phone}/decomposition_summary.csv`
  (+ per-model CSVs, `projection_results.json`, timestamped logs).

## How to run

```bash
git pull
./run_projection_analysis.sh          # grouped split by utt_id (default)
# if MLP R² stays ~1.0, hold out types instead:
#   add  --group_by word   (word level)  /  --group_by phone  (phone level)
```

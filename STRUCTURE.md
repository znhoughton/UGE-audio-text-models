# Repository structure

Reorganized (Sep 2026) around the **"shares information, not geometry"** analysis.
Large embedding/audio caches (`*.pkl`, `*.npy`, `PhoneData/`, `WordData/`, …) are
gitignored and left in place.

```
extraction/     Shared embedding-extraction engine (run these from inside extraction/)
  representation_analysis.py     per-model extractors (whisper enc/dec, kaldi, wav2vec2, parakeet, mimi, LLMs)
  word_level_analysis.py         word-level extraction + helpers (CKA, effective rank, loaders)
  phone_level_analysis.py        phone-level driver (calls the extractors)
  whisper_layer_similarity.py    PER-LAYER extraction (writes PhoneLayerData/layer_embeddings/)
  projection_analysis.py         variance decomposition / predictivity engine (both granularities)
  run_projection_analysis.sh     launcher for projection_analysis.py

phone/          Phone-level analyses (the paper's spine)
  layer_trajectory.py            GEOMETRY: per-layer mutual-kNN to acoustic/word-identity anchors, across sizes
  info_vs_geometry.py            CAPSTONE: information (ridge R2) vs geometry (mutual-kNN) per layer
  plot_info_vs_geometry.py       renders phone/figures/info_vs_geometry_by_layer.png
  results/                       result JSONs (info_vs_geom, layer_traj, projection, residual/rogue/subspace diagnostics)
  figures/                       info_vs_geometry_by_layer.png (money plot) + phone CKA/eigenspectra
  PROGRESS.md                    projection-analysis notes

word/           Word-level analyses (LLM comparison is honest at word scale)
  geometry_mutual_knn.py         GEOMETRY: mutual-kNN alignment matrix across models
  results/                       word JSONs (mutual-kNN/cknna, rsa, granularity, projection)
  figures/                       word CKA/eigenspectra

archived/       Superseded / other-corpus material (kept for history)
  scripts/      old analyses (MCV/MLS cross-speaker, deaf-decoder, old variance_decomposition, tests, plotting)
  plots/        MCVPlots, MLSPlots, DecompositionPlots, old Plots/, DecompositionData
  old_logs/     projection run logs
```

**Environment:** `/opt/modeling/zhoughton/envs/propsensity_eval/bin/python` (pyenv 3.12.4;
the default pyenv shim is glibc-broken). GPU (per-layer extraction) only works when
launched OUTSIDE the Claude Code sandbox.

**Note:** several exploratory scratchpad scripts (CKNNA, subspace, rogue, residual,
fair-interaction, granularity, rsa) were lost to pod restarts; their **result JSONs are
preserved** in `phone/results/` and `word/results/`. `word/geometry_mutual_knn.py` was
reconstructed. The core capstone scripts (`phone/info_vs_geometry.py`,
`phone/layer_trajectory.py`) survived intact.

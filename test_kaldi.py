#!/usr/bin/env python3
"""
Smoke test for the Kaldi nnet3 acoustic model extraction pipeline.

Verifies that:
  1. torchaudio.compliance.kaldi computes MFCC features without error
  2. nnet3-compute runs successfully via subprocess
  3. kaldiio reads the output ark correctly
  4. The output shape and FPS match expectations (~199 frames, ~100 Hz for 2s audio)

Usage:
  python test_kaldi.py \
    --kaldi_bin /opt/modeling/nsharma/kaldi/src/nnet3bin \
    --model_dir exp/chain_cleaned/tdnn_1d_sp \
    --output_node prefinal-chain.batchnorm2
"""

import argparse
import tempfile
import numpy as np
import torch
import torchaudio
import subprocess
import kaldiio
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kaldi_bin", default="/opt/modeling/nsharma/kaldi/src/nnet3bin", type=Path)
    p.add_argument("--model_dir", default="exp/chain_cleaned/tdnn_1d_sp", type=Path)
    p.add_argument("--output_node", default="prefinal-chain.batchnorm2", type=str)
    p.add_argument("--ivector_dim", default=100, type=int)
    p.add_argument("--duration_s", default=2.0, type=float,
                   help="Duration of synthetic test audio in seconds")
    args = p.parse_args()

    model_path = args.model_dir / "final.mdl"
    nnet3_bin  = args.kaldi_bin / "nnet3-compute"

    for path in [model_path, nnet3_bin]:
        if not path.exists():
            raise FileNotFoundError(f"Not found: {path}")

    sample_rate = 16000
    n_samples = int(args.duration_s * sample_rate)
    audio = (np.random.randn(n_samples) * 0.01).astype(np.float32)
    print(f"Test audio: {args.duration_s}s @ {sample_rate} Hz ({n_samples} samples)")

    # MFCC features (matches Kaldi's conf/mfcc_hires.conf for chain models)
    waveform = torch.from_numpy(audio).unsqueeze(0)
    feats = torchaudio.compliance.kaldi.mfcc(
        waveform,
        sample_frequency=float(sample_rate),
        num_ceps=40,
        num_mel_bins=40,
        low_freq=20.0,
        high_freq=-400.0,
        frame_shift=10.0,
        frame_length=25.0,
        dither=0.0,
        snip_edges=True,
        use_energy=False,
    ).numpy().astype(np.float32)
    feats -= feats.mean(axis=0, keepdims=True)
    print(f"MFCC features: {feats.shape}  (expect ~{n_samples // 160}, 40)")

    IVEC_PERIOD = 10
    n_chunks = (feats.shape[0] + IVEC_PERIOD - 1) // IVEC_PERIOD
    ivec = np.zeros((n_chunks, args.ivector_dim), dtype=np.float32)

    with tempfile.TemporaryDirectory(prefix="kaldi_test_") as tmp:
        tmp = Path(tmp)
        with kaldiio.WriteHelper(f"ark,scp:{tmp}/mfcc.ark,{tmp}/mfcc.scp") as w:
            w["utt1"] = feats
        with kaldiio.WriteHelper(f"ark,scp:{tmp}/ivec.ark,{tmp}/ivec.scp") as w:
            w["utt1"] = ivec

        cmd = [
            str(nnet3_bin),
            "--use-gpu=no",
            f"--online-ivectors=scp:{tmp}/ivec.scp",
            f"--online-ivector-period={IVEC_PERIOD}",
            f"--output-node={args.output_node}",
            "--apply-exp=false",
            str(model_path),
            f"scp:{tmp}/mfcc.scp",
            f"ark:{tmp}/hidden.ark",
        ]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            print("FAILED — nnet3-compute stderr:")
            print(result.stderr[-2000:])
            raise SystemExit(1)

        hidden_dict = dict(kaldiio.load_ark(str(tmp / "hidden.ark")))
        if "utt1" not in hidden_dict:
            print("FAILED — nnet3-compute produced no output for utt1")
            raise SystemExit(1)

        hidden = hidden_dict["utt1"]
        fps = hidden.shape[0] / args.duration_s
        print(f"SUCCESS")
        print(f"  Output shape : {hidden.shape}  (expect ~{n_samples // 160}, 256)")
        print(f"  FPS          : {fps:.1f}  (expect ~100.0)")
        print(f"  Output node  : {args.output_node}")
        if abs(fps - 100.0) > 5:
            print(f"  WARNING: FPS {fps:.1f} is far from 100 — check --output_node or "
                  "update KALDI_FPS constant in word_level_analysis.py")


if __name__ == "__main__":
    main()

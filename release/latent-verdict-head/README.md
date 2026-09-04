# Latent Verdict Head (LVH)

**Single-pass process verification from a model's intermediate representations.**

> Verifying whether a reasoning step is correct does not require generating a
> critique or executing code — the judgment is *already* present in the
> verifier's hidden states. A lightweight **residual verdict head** (mid-layer +
> final-layer) fine-tuned with LoRA reads it out in **one forward pass**,
> matching generate-and-execute process reward models (PRMs) at a small fraction
> of the compute (single V100-class GPU, a few hours).

## Method

For each solution step we take the last-token hidden state at two depths and sum
two small heads:

```
logit(step) = Linear(h_last)  +  MLP(h_layer-9)
reward      = sigmoid(logit)          # > threshold  ⇒  step is an error
```

- `h_last` — final layer, optimized to *produce the verbalized `Yes/No` token*
  ("what the model says"; presentation-oriented, biased by fine-tuning and by
  fluent-but-wrong steps).
- `h_layer-9` — a mid-upper layer (~70% depth; best by a per-layer sweep), the
  model's **assessment before it is compressed into the surface token**
  ("the deliberation behind the answer").
- The **residual sum** fuses *stated* and *latent* verdicts; it beats either
  view alone and breaks the frozen-probe ceiling on proof domains.

The backbone is LoRA fine-tuned (r=16, q/k/v/o) on step-level `Yes/No` labels.
Discriminative — **no generation, no code execution, one forward pass.**

## Results (ProcessBench, localization F1)

**1.5B — full ProcessBench** (DeepSeek-R1-Distill-Qwen-1.5B backbone):

| method | gsm8k | math | olympiad | omnimath | MEAN | compute |
|---|---|---|---|---|---|---|
| GenPRM-1.5B (generate+code) | 52.8 | 66.6 | 55.1 | 54.5 | 57.2 | ~20–50× |
| **LVH-1.5B (this repo)** | 65.1 | 61.3 | 53.5 | 52.4 | **58.1** | **1×** |
| **LVH ⊕ GenPRM fusion** | — | — | — | — | **+5.4** over GenPRM | 1× + PRM |

LVH matches GenPRM **on average at ~1/20–1/50 the cost**, and its signal is
*complementary*: honest dual-view fusion (weights tuned on a held-out MATH set,
`fuse_1p5b.py`) adds **+5.4** mean F1 over GenPRM on 120-case slices.

**7B — 120-case slices** (DeepSeek-R1-Distill-Qwen-7B backbone):

| method | gsm8k | math | olympiad | omnimath | MEAN |
|---|---|---|---|---|---|
| GenPRM-7B | 83.4 | 80.0 | 72.3 | 71.5 | 76.8 |
| LVH-7B standalone (peak) | ~70 | ~70 | ~55 | ~55 | ~62 |
| **LVH ⊕ GenPRM-7B, domain-calibrated, honest fusion** | 83.4 | 81.5 | 73.2 | 71.5 | **77.4** |

At 7B the generative PRM is strong enough that the cheap head no longer matches
it alone; but with **unsupervised per-domain calibration** (`calibrate_fuse_7b.py`,
strategy *B* = per-domain median-centering of the head logits) the honest
leave-one-domain-out fusion still **improves every domain** (+0.6 mean, no domain
below GenPRM). This gives a practical cost–accuracy rule: LVH is Pareto-optimal
at small scale; at large scale it is a near-free add-on to a generative PRM.

### Honest notes
- The per-domain calibration is a **scale-specific** fix: it helps the 7B head
  (which develops an olympiad calibration offset) but *hurts* the 1.5B head
  (`calibrate_1p5b.py`, MEAN 58.1→55.9) — a finding, not a universal trick.
- `omnimath` is a genuine ranking limit for the cheap head (step-AUC ≈ 0.57);
  fusion correctly defers to the generative PRM there (`diag_omnimath.py`).

## Repository layout

| file | role | key result |
|---|---|---|
| `train_lvh_1p5b.py` | **1.5B non-fusion** — train LVH + eval full ProcessBench | 58.1 |
| `train_lvh_7b.py` | **7B non-fusion** — train LVH (periodic eval + atomic checkpoint/resume) | ~62 peak |
| `fuse_1p5b.py` | **1.5B fusion** — LVH ⊕ GenPRM, weights tuned on held-out MATH | +5.4 |
| `dump_scores_7b.py` | dump LVH `P_A` + GenPRM `P_B` + labels to `.npz` (7B) | — |
| `calibrate_fuse_7b.py` | **7B fusion (best)** — per-domain calibration + honest LODO fusion | 77.4 |
| `fuse_7b_direct.py` | 7B fusion without calibration (baseline for the above) | 76.6 |
| `layer_sweep_7b.py` | per-layer probe sweep → justifies layer-9 | AUC/locF1 by layer |
| `dump_scores_1p5b.py` | dump LVH `P_A` + labels to `.npz` (1.5B, full ProcessBench) | — |
| `calibrate_1p5b.py` | 1.5B calibration study (negative transfer of strategy B) | 58.1→55.9 |
| `diag_omnimath.py` | diagnosis of the omnimath ranking limit | step-AUC, length strata |

## Data / environment

Scripts expect this working directory layout (paths are relative):

```
models/DSR1-1.5B , models/DSR1-7B         # DeepSeek-R1-Distill-Qwen bases (HF)
GenPRM-Data/data/train-00000-of-00001.parquet   # step-level Yes/No labels (training)
pb_<domain>_full_in/<case>/sample.json    # ProcessBench cases (problem, steps, label)
pb_<domain>_7b_out/<case>/result_1.json   # GenPRM-7B outputs; 'value' = P_B
fthead_lora/ , fthead_heads.pt            # trained 1.5B LVH (produced by train_lvh_1p5b.py)
ck7b*.pt                                   # 7B LVH checkpoints (produced by train_lvh_7b.py)
```

Dependencies: `torch`, `transformers`, `peft`, `pyarrow`, `numpy` (a
`bitsandbytes`-free bf16/fp16 setup fits a single 32 GB GPU; 1.5B fits a V100).

## Typical run

```bash
# 1.5B non-fusion: train + evaluate
python train_lvh_1p5b.py 8000 15000            # n_convs, n_steps  → fthead_lora, fthead_heads.pt

# 1.5B fusion with a generative PRM
python fuse_1p5b.py                            # tunes (alpha,bias) on MATH-val, reports vs GenPRM

# 7B non-fusion: train (checkpoints every 500 steps, resumable)
python train_lvh_7b.py 8000 12000

# 7B fusion (best): dump scores once, then iterate calibration in numpy
python dump_scores_7b.py ck7b_4000.pt pa_7b.npz 4096
python calibrate_fuse_7b.py pa_7b.npz          # strategies A/B/C + honest LODO fusion
```

*Training data are GenPRM-Data step labels (reproducible math-model trajectories
with relative-progress estimation); the method itself does not require the GenPRM
model — only its released step labels for supervision.*

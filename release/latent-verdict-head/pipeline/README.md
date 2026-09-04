# Self-Configuring LVH Pipeline (base model only, no GenPRM)

This pipeline builds a Latent Verdict Head from a base model alone. It selects the readout layer
by having the base model write and run its own selection code, then trains the residual head at
that layer with LoRA. No generative PRM (GenPRM) is used at any stage; only step-level labels are
reused for supervision.

## Stages

| stage | script | input | output |
|---|---|---|---|
| 1. Feature dump | `stage1_dump_layer_feats.py` | base model, labeled step data | `layerfeats.npz` (per-layer step-boundary features + first-error labels) |
| 2. Self-written selector | `stage2_generate_selector.py` | `layerfeats.npz`, `SKILL_select_layer.md` | `model_selector.py` (base-model code), `selected_layer.txt` |
| 3. Reference / fallback | `ref_selector.py` | `layerfeats.npz` | deterministic layer choice (verification, or fallback if Stage 2 fails) |
| 4. Train LVH | `stage4_train_lvh.py` | `selected_layer.txt`, base model, step labels | `fthead7b_lora/`, `fthead7b_heads.pt` |

Run everything with `bash run_pipeline.sh` (see the script for `NCONV`/`NSTEPS`/`PY` overrides).

## How Stage 2 works

The base model is given `SKILL_select_layer.md`, a detailed recipe that specifies the exact data
interface, the algorithm (per-layer logistic-regression probe ranked by localization F1), the common
pitfalls, and the output format. The model writes a complete Python script; `stage2_generate_selector.py`
extracts it, runs it, and on success records the chosen layer in `selected_layer.txt`. The step is a
plain sampler: it retries with fresh samples until one script runs and prints `SELECTED = X`.

## Validation (what we observed)

- A distilled 7B reasoning base model **cannot** write a correct selector from a terse prompt or by
  iterating on its own error traces. With the detailed skill it writes the correct algorithm; the
  remaining failures are execution-harness issues (loading object arrays without `allow_pickle=True`,
  extracting the wrong code block, an over-tight timeout). Once those are handled, the model's code
  runs on the first sample and matches an independent reference selector exactly.
- The selected layer sits in a mid-upper **plateau** (roughly 70-80% depth). The exact index is
  sample dependent: on one split the argmax is layer -7, on another (closer to the original scale) it
  is layer -9, and the base-model code reproduces the reference on both. The method locates the
  optimal band rather than one fixed index, so it is robust to the precise choice.

Read this as **skill-guided self-configuration**: the base model configures its own readout layer,
given a sufficiently detailed skill and a robust execution wrapper. It is not the model writing
correct numerical code unaided.

## Expected data layout

```
models/DSR1-7B                              # base model (DeepSeek-R1-Distill-Qwen-7B)
GenPRM-Data/data/train-00000-of-00001.parquet   # step-level Yes/No labels (supervision only)
pb_<domain>_7b_out/<case>/result_1.json     # 120-case eval slices used by Stage 4
```

Dependencies: `torch`, `transformers`, `peft`, `pyarrow`, `numpy`, `scikit-learn`.

#!/usr/bin/env bash
# Self-configuring LVH pipeline (base model only, no GenPRM).
# Stage 1: extract per-layer step-boundary features on a labeled validation set.
# Stage 2: the BASE model writes the layer-selection script (guided by SKILL_select_layer.md),
#          which is executed; the chosen layer is written to selected_layer.txt.
# Stage 3: (fallback / verification) a deterministic reference selector, in case Stage 2 fails.
# Stage 4: LoRA + residual verdict head trained at the selected mid layer.
#
# Paths are relative to the working directory (see repo README for the expected layout:
# models/DSR1-7B, GenPRM-Data/, pb_*_7b_out/). Run with the project's Python env.
set -e
PY=${PY:-python}
NCONV=${NCONV:-8000}
NSTEPS=${NSTEPS:-12000}

echo "== Stage 1: dump per-layer features =="
$PY stage1_dump_layer_feats.py 560 140          # -> layerfeats.npz

echo "== Stage 2: base model writes + runs the layer selector =="
if $PY stage2_generate_selector.py 8; then
  echo "selected layer: $(cat selected_layer.txt)"
else
  echo "Stage 2 did not converge; falling back to the reference selector."
  $PY ref_selector.py | tee ref_out.txt
  grep -oE 'SELECTED = -?[0-9]+' ref_out.txt | grep -oE '\-?[0-9]+' > selected_layer.txt
  echo "selected layer (reference): $(cat selected_layer.txt)"
fi

echo "== Stage 4: train LVH (LoRA + residual head) at the selected layer =="
$PY stage4_train_lvh.py $NCONV $NSTEPS            # reads selected_layer.txt

echo "ALL_STAGES_DONE"

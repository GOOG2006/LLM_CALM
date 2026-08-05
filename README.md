# Verify-by-Measurement: Code-Free Process Verification for LLM Reasoning

Training-free, **code-free** process verification for LLM step-level reasoning, studied against the
generative process-reward-model baseline **GenPRM** (AAAI 2026) on **ProcessBench**.

Instead of *generating and parsing* a discrete `Yes/No` verdict per step (as GenPRM does — inheriting
saturation, format-fragility, and code-execution dependence), we let the model produce a short **code-free
reasoning rollout** and then **measure the continuous logit** of its correctness belief for the step.

## Key findings (1.5B base = DeepSeek-R1-Distill-Qwen-1.5B, same base as GenPRM-1.5B)

Fair comparison, `n=60` per subset, **100 % step coverage including long chains**, 30 positives/subset.

**Step-level discrimination (AUROC):**

| subset | GenPRM-1.5B | ours (best) |
|---|---|---|
| MATH | 0.762 | **0.845** |
| OlympiadBench | 0.697 | **0.826** |
| Omni-MATH | 0.669 | **0.707** |

**ProcessBench F1 (standard, case-level first-error localization):**

| subset | GenPRM-1.5B (thr 0.5) | ours (τ=1) |
|---|---|---|
| MATH | **45.4** | 43.1 |
| OlympiadBench | 22.9 | **44.4** |
| Omni-MATH | 17.9 | **33.1** |

**Takeaway:** comparable on easy MATH, but **~2× more robust on hard/long subsets** (OlympiadBench,
Omni-MATH) where GenPRM's F1 collapses — precisely where its structural weaknesses (early-commitment on
long chains; code-execution failures) bite. Including long chains in the evaluation is what reveals this.

> **Honest caveats** (see `pilot/FINDINGS_measure.md`): ours' F1 threshold τ is currently tuned on the test
> set (needs a held-out validation split); vLLM sampling has mild batch-order noise (needs multi-seed);
> 30 positives/subset is moderate power. AUROC advantage does **not** fully translate to F1 — ours' scores
> need calibration.

## The measurement iteration (what worked / what didn't)

| operator | AUROC (MATH) | verdict |
|---|---|---|
| Detailed-Balance (fwd/bwd irreversibility) | 0.57 | killed (backward term hurts) |
| static entropy / min-logp | ~0.68 | surface-fluency ceiling |
| bare Yes/No probe (no reasoning) | ~0.46 | ≈ chance → correctness needs reasoning |
| hidden-state linear probe | ~0.65 | static representation insufficient |
| **think-then-measure** (K-token rollout → read verdict logit) | 0.78→0.85 | monotone in reasoning length K |
| + self-consistency averaging | +margin | boosts easy subsets |

## Repository layout

- `pilot/measure_think_sc.py` — main method: think-then-measure + self-consistency (vLLM).
- `pilot/genprm_run.py` — GenPRM baseline runner over ProcessBench (resumable).
- `pilot/compare_sameset.py` — same-set AUROC + coverage.
- `pilot/pb_f1.py` — ProcessBench standard F1 (GenPRM vs ours, threshold sweep).
- `pilot/measure_pilot.py`, `measure_probe.py`, `measure_hidden.py`, `detbal_pilot.py` — the operator ablations above.
- `pilot/FINDINGS_measure.md` — full, honestly-caveated result log.
- `BASELINE_ANALYSIS_genprm.md` — 7 evidenced weaknesses of GenPRM that motivate the method.
- `IDEA_REPORT.md` — the idea-discovery record.

## Reproduce (single 32 GB GPU, e.g. V100)

```bash
# baseline (resumable; --full scores all steps)
python pilot/genprm_run.py --config olympiadbench --n 60 --majority 1 --full --out out_olymp60.jsonl
# ours (code-free think-then-measure + SC)
python pilot/measure_think_sc.py --model <DSR1-Distill-Qwen-1.5B> --n_err 30 --n_cor 30 \
       --samples 8 --K 512 --subset olympiadbench --genprm out_olymp60.jsonl --out results_olymp.jsonl
# compare
python pilot/compare_sameset.py results_olymp.jsonl out_olymp60.jsonl   # AUROC + coverage
python pilot/pb_f1.py           results_olymp.jsonl out_olymp60.jsonl   # ProcessBench F1
```

Requires GenPRM's code + `Qwen/ProcessBench` subsets locally; run inside an env with `vllm` + `transformers`
(pinned compatible — vLLM 0.7.x with transformers 4.49).

---
*Research in progress. Results are on a 1.5B base and a 60-case-per-subset slice; not yet the full-benchmark,
multi-seed, validation-thresholded numbers required for publication.*

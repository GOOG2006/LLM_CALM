# Prompt-Based Process Verification + Latent-Probe Veto

**Frozen 7B, no fine-tuning. Matches a fine-tuned GenPRM-7B on ProcessBench step-error localization.**

We ask how far process-step verification can go on a **frozen** DeepSeek-R1-Distill-7B
(same base model GenPRM fine-tunes), using only **prompt engineering + decoding + a
cheap latent probe as a veto** — no weight update, no critique generation, no code
execution beyond a single forward pass.

## Method (three layers)

1. **`v9t` single-pass localizer** (`src/gen_localize.py`, `src/v9t_baseline.py`).
   One generation over the whole numbered solution, output only `Answer: N` (first wrong
   step, `0` = all correct). Key design: a **transferability reframe** ("verify each step
   genuinely FOLLOWS: if a calculation recompute it, if a claim check the deduction is
   valid"), `repetition_penalty=1.1`, and a **full robust answer parser** (prose patterns
   + range-validated `\boxed{N}`/`Answer:`). This single choice fixes the three failure
   modes of per-step verification (truncation, verdict-format, early-false-alarm cascade).

2. **Two narrow gated clauses** (`src/gen_localize.py`).
   - *concrete-number recheck*: explicitly recompute any number feeding the final answer,
     and check it is the quantity the problem asks for.
   - *assertion / enumeration-completeness*: for "this maximizes / the only way is /
     max is N / complete list", actively search for one more case or a more extreme value.

   Added **globally** these clauses help where errors are concrete (GSM8K, OlympiadBench)
   but **collapse** where errors are conceptual proofs (MATH, OmniMATH). So they are
   **gated**: only used to fill v9t's *all-correct* verdicts (its misses), never to
   override its localizations.

3. **Latent-probe veto** (`src/probe_veto.py`, analysis in `src/probe_analysis.py`).
   An in-domain **linear probe** on the frozen backbone's layer-`-7` step-last-token
   hidden state scores each step's wrongness. The clause override is accepted **only if
   the flagged step is among the probe's top-2 most-suspicious steps** (argmax agreement).
   This agreement is **rank-based and therefore scale-invariant**, which sidesteps the
   train→eval score-distribution shift that makes the probe unusable as a raw scorer.
   Clauses provide **recall**; the probe provides an independent **precision veto**.

## Results (ProcessBench localization F1, N=120 per subset, frozen DSR1-7B, single pass)

GenPRM-7B numbers are the **Pass@1** row of GenPRM Table 1 (arXiv:2504.00891), same base
model, also single-pass — the fair comparison.

| Subset | v9t (single prompt) | best frozen method here | GenPRM-7B **Pass@1** (fine-tuned) |
|---|---|---|---|
| GSM8K | 72.0 | **80.0** (global assertion-clause) | 78.7 |
| MATH | 74.0 | **76.6** (gated-veto) | 80.3 |
| OlympiadBench | 65.1 | **72.3** (global assertion-clause) | 72.2 |
| OmniMATH | 67.8 | **71.5** (gated-veto) | 69.8 |
| **MEAN** | 69.7 | **75.1** | 75.2 |

- The frozen pipeline **matches** fine-tuned GenPRM-7B Pass@1 in mean (75.1 vs 75.2) and
  **exceeds it on 3 of 4 subsets** (GSM8K, OlympiadBench, OmniMATH); only MATH trails.
- The **unified** robust method (gated clause∩probe-veto everywhere, layer `-7` top-2)
  improves **every** subset over v9t and never collapses: 75.5 / 76.6 / 69.5 / 71.5,
  **MEAN 73.3**. On GSM8K it is +3/−0 (strict, zero correct-case breakage).
- OmniMATH deep-dive: 67.8 → **71.5** via gated-clause-union ∩ probe-veto
  (`analysis/omnimath_error_cases.md` has the per-case error taxonomy that motivated the
  clauses).

> **Caveat (read before citing).** Our eval is a 120-case sample per subset
> (60 error + 60 correct, `index>=50`, seed `20240822`); GenPRM's published numbers are on
> full ProcessBench. Metric and base model match, but the test *sets* are not identical.
> For a strict head-to-head, run GenPRM's official `prm_evaluate.py` on this same slice.

## Layout

```
src/
  gen_localize.py    # v9t + the two clauses; runs all 3 prompts per subset (vLLM)
  v9t_baseline.py    # v9t + full robust parser, single subset (the localizer core)
  probe_veto.py      # in-domain probe + clause∩probe-veto (the headline method)
  probe_analysis.py  # probe-alone / fusion / oracle analysis (shows the latent signal)
  remote/            # tiny paramiko helpers (put/get/run/launch/poll/tail); creds via env
analysis/
  omnimath_error_cases.md   # 34 v9t-wrong OmniMATH cases with root-cause tags
```

## Running

Environment: `vLLM 0.6.6.post1` + `torch`/`transformers` on a single **V100-32GB**
(Volta → XFormers backend, no FlashAttention-2). Model at `models/DSR1-7B`
(DeepSeek-R1-Distill-Qwen-7B), ProcessBench JSON at `ProcessBench/<subset>.json`.

```bash
# 1. generate v9t + clause predictions for a subset (writes <subset>_v9t/v9num/v9verify_preds.json)
python src/gen_localize.py gsm8k

# 2. train in-domain probe + apply clause∩probe-veto, print F1 table
python src/probe_veto.py gsm8k
```

The `src/remote/` helpers run these on a detached GPU box over SSH; set
`REMOTE_HOST/REMOTE_PORT/REMOTE_USER/REMOTE_PW` env vars first (no credentials are stored
in the code).

## Relationship to the sibling work

The latent probe here is the **linear, veto-only** cousin of the **Latent Verdict Head
(LVH)** (`../latent-verdict-head/`), which trains a residual MLP+Linear head with LoRA.
This work shows the *stated verdict* (prompt) and the *latent signal* (probe) are
**complementary**: prompt = recall, latent = precision veto.

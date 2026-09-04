# Release — two independent works

Each subfolder is **self-contained** and can be uploaded as its own GitHub repository.
They do not share files.

## 1. `latent-verdict-head/` — Introspective Verification (LVH)  *[Paper 1, has paper]*

Step correctness is already encoded in a frozen verifier's **hidden states**. A small
**residual verdict head** (intermediate layer `-9` MLP + final-layer Linear), LoRA-adapted
on step labels, reads it out in **one forward pass** — no critique generation, no code.
Trains and runs on a single **V100**. On ProcessBench (1.5B backbone) it reaches 58.1 mean
F1, exceeding most PRMs including 7B/14B trained ones.

- Code: top level (`train_lvh_*.py`, `fuse_*.py`, `calibrate_*.py`, `pipeline/`).
- Paper: `paper/` (LaTeX source + `main.pdf`), title *"Introspective Verification:
  Reading Step Correctness at Intermediate Depth"*.

## 2. `prompt-process-verification/` — Prompt + Latent-Probe Veto  *[Paper 2, code only, no paper yet]*

How far a **frozen** DeepSeek-R1-Distill-7B goes on ProcessBench with only prompt design +
a cheap **latent probe used as a veto** (no fine-tuning). A single-pass localizer (`v9t`)
plus two gated narrow clauses, with the clause override accepted only when an in-domain
linear probe agrees (top-2, rank-based → scale-invariant). **Matches fine-tuned GenPRM-7B
Pass@1 in mean (75.1 vs 75.2) and beats it on 3 of 4 subsets**, frozen.

- Code: `src/` (method) + `src/remote/` (SSH helpers, credentials via env vars).
- Analysis: `analysis/omnimath_error_cases.md`.
- No paper drafted yet.

---

### Shared context
- Base model: **DeepSeek-R1-Distill-Qwen** (1.5B / 7B). GPU: single **V100-32GB** (Volta;
  vLLM uses the XFormers backend, FlashAttention-2 is unavailable on Volta).
- Benchmark: **ProcessBench** (GSM8K / MATH / OlympiadBench / OmniMATH), F1 of
  error-step-accuracy and correct-solution-accuracy.
- Reference upper bound: **GenPRM** (arXiv:2504.00891), a *fine-tuned* generative PRM on the
  same base model. Comparisons use its **Table 1 Pass@1** row (single-pass, the fair match):
  GSM8K 78.7 / MATH 80.3 / OlympiadBench 72.2 / OmniMATH 69.8 / **Avg 75.2**.

> **Note on one number in the LVH paper.** `latent-verdict-head/paper` currently uses
> `76.8` as a generative-model mean baseline. `76.8` is actually the **OmniMATH Maj@8**
> cell of GenPRM Table 1, not a Pass@1 mean (the Pass@1 mean is 75.2). Worth re-checking
> that fusion sentence before submission.

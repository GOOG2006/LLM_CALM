# Introspective Verification: Reading Step Correctness at Intermediate Depth

Code for the submission *Introspective Verification: Reading Step Correctness at
Intermediate Depth*.

Process reward models score whether each reasoning step is correct. The most accurate
ones do this by generating a critique and often calling an interpreter, which makes
checking a solution more expensive than producing it. This work shows that much of the
same judgment is already present in the verifier backbone's hidden states, and that a
small residual readout recovers it in a single forward pass, with no generation and no
code execution.

## Code

All released code is under [`release/latent-verdict-head/`](release/latent-verdict-head/).

| Script | Purpose | Paper |
| --- | --- | --- |
| `layer_sweep_7b.py` | Per layer probe sweep that selects the intermediate layer | Alg. 2, Tab. 2 |
| `train_lvh_1p5b.py` / `train_lvh_7b.py` | Train the residual verdict head with LoRA | Sec. 3.5 |
| `dump_scores_1p5b.py` / `dump_scores_7b.py` | Write per step scores for every evaluation case | Sec. 4 |
| `calibrate_1p5b.py` / `calibrate_fuse_7b.py` | Decision threshold and the per domain shift used at 7B | Sec. 4.5, App. B.4 |
| `fuse_1p5b.py` / `fuse_7b_direct.py` | Logit space fusion with a generative verifier | Sec. 3.7, Sec. 4.6 |
| `diag_omnimath.py` | OmniMATH ranking analysis | App. B.5 |

See [`release/latent-verdict-head/README.md`](release/latent-verdict-head/README.md) for
the method summary, environment, and run order.

## Setup

Training and inference both run on a single 32GB V100. The backbone is
DeepSeek-R1-Distill-Qwen at 1.5B and 7B. Evaluation is ProcessBench localization F1.
Step labels come from the released GenPRM label set, drawn from MATH training problems
only. No ProcessBench case contributes a training label.

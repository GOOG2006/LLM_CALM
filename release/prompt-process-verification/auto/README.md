# Auto-Framework: error-driven, self-improving prompt optimization

Automates the human loop that produced the hand-built clauses. An analyst LLM reads the
localizer's mistakes, tags the *checkable* error families, writes a narrow clause per
family, and the framework keeps a clause only if it is net-positive under three safety
invariants — the invariants are what make automatic clause-adding safe where generic
auto-prompt-optimization overfits or collapses.

```
run localizer ─▶ dump wrong (dev) ─▶ Tagger: {failure_mode, error_family, checkable}
      ▲                                     │ cluster the CHECKABLE families
      │                                     ▼
  clause library ◀── keep iff net>0 ── GATED eval (val) ◀── Proposer: one narrow clause
      │  (grows monotonically)
      ▼
  UNION (prediction-level)  ∩  PROBE-VETO (latent top-k)  ─▶ report on held-out test
```

## The three safety invariants (enforced in `loop.py` / `combiner.py`)

1. **Gate** — a new clause may only fill the localizer's *all-correct* verdicts (its
   misses); it can never override a localization. Adding a clause is then near-monotone.
2. **Union, not merge** — clauses run as separate passes and are combined at the
   prediction level. Merging two clauses into one prompt makes them interfere and collapse
   (measured: single-prompt merge 39.3 vs union 69.6).
3. **Probe-veto** — a clause override survives only if an in-domain latent probe ranks that
   step in its top-k most-suspicious (rank-based ⇒ scale-invariant ⇒ immune to the
   train→eval score shift that makes the probe useless as a raw scorer).

The **`checkable` flag** from the Tagger is the other key idea: clause proposals are spent
only on families a mechanical check can catch (arithmetic / enumeration / definition), not
on conceptual proof errors (the frozen-model capability wall — no clause helps there).

## Files

| file | role | needs |
|---|---|---|
| `llm_backend.py` | analyst LLM (OpenAI-compatible **or** local-7B), text-in/out | API key *or* GPU |
| `tagger.py` | tag each wrong case → `{failure_mode, error_family, checkable}`; cluster | analyst LLM |
| `proposer.py` | write ONE narrow clause from a cluster (+ negative controls) | analyst LLM |
| `combiner.py` | gate / union / probe-veto / F1 / rescue-broke — **pure CPU** | — |
| `loop.py` | the closed loop + accept rule + collapse guard | GPU verifier + analyst |
| `replay_demo.py` | **CPU-only** replay of the two known clauses → reproduces 69.6 | — |
| `replay_verifier.py` | cached "verifier" so `loop.py --replay` runs off disk | — |
| `sample_data/` | cached OmniMATH clause predictions used by the demo | — |

## What needs a GPU vs CPU

- **CPU only**: `combiner.py`, `replay_demo.py`, all the gate/union/veto logic, and the
  Tagger/Proposer *if* you point `ANALYST_BACKEND=openai` at an external API.
- **GPU (V100)**: every time a *newly proposed* clause is tested — `loop.py` calls the
  frozen DSR1-7B on the eval set — and probe scoring. There is no way around this: testing
  a new prompt means running the verifier.

## Run

```bash
# CPU-only demo (no GPU, no API): validate the safe-combination machinery
python replay_demo.py
#   localizer 67.8 -> +concrete-number 69.2 -> +assertion 68.1 -> union 69.6
#   (+probe-veto -> 71.5 once sample_data/omnimath_probe_scores_L-7.json exists)

# Full closed loop (needs GPU verifier + analyst LLM):
export ANALYST_BACKEND=openai OPENAI_API_KEY=... OPENAI_MODEL=gpt-4o
python loop.py            # wire a live Verifier in loop.py (see the Verifier ABC)
```

## Honest ceiling

The loop automates the *labor* of finding clauses, not model capability. Its reachable
ceiling is the same as the manual process (~75 mean across ProcessBench): it recovers the
**checkable** misses and leaves the **conceptual** proof errors — those need a stronger base
model or a trained head (see `../../latent-verdict-head/`), not another clause.

## Relation to prior auto-prompt work

Same family as OPRO / APE / EvoPrompt / Reflexion (LLM proposes prompts, feedback selects).
The contribution here is the **safety layer** (gate + union + latent probe-veto) plus the
**checkable/conceptual split**, which together stop the overfitting and precision-collapse
that sink generic auto-prompt-optimization on process verification.

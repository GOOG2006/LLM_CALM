# -*- coding: utf-8 -*-
"""Proposer: given a cluster of same-family CHECKABLE errors (positives) plus a few
CORRECT solutions (negatives), write ONE narrow clause to append to the localizer prompt.

Two objectives are baked into the instruction, matching what actually worked by hand:
  1. catch this specific error family, and
  2. do NOT fire on the correct (negative) solutions.
The clause is deliberately narrow; it will be added in GATED mode (fill misses only), so
over-firing is contained, but low false-alarm rate still matters for the probe-veto stage.
"""
from llm_backend import chat

PROPOSE_PROMPT = """We localize the first wrong step of math solutions with a frozen model.
It systematically MISSES one family of errors. Write ONE short instruction clause (2-4
sentences) to append to the checker prompt so it catches this family, WITHOUT flagging
valid steps.

Error family: {family}

Examples the checker WRONGLY passed (the real first wrong step is quoted):
{positives}

Correct solutions it must NOT start flagging (negative controls):
{negatives}

Requirements for the clause:
- Target ONLY this family with a concrete, mechanical check the model can actually perform
  (recompute a number, search for one more case, re-read a definition/target quantity).
- Explicitly say it applies only to this situation, not to conceptual/proof claims.
- End effect: "if the check fails, THAT step is the first wrong step."
Return ONLY the clause text, no preamble."""


def _fmt_pos(cases):
    out = []
    for c in cases[:5]:
        e = c["label"]
        step = c["steps"][e].strip()[:280] if 0 <= e < len(c["steps"]) else "(all-correct)"
        out.append(f"- true first-wrong step: {step}")
    return "\n".join(out)


def _fmt_neg(cases):
    out = []
    for c in cases[:3]:
        out.append("- " + " | ".join(s.strip()[:90] for s in c["steps"][:3]))
    return "\n".join(out)


def propose_clause(family, positive_cases, negative_cases):
    p = PROPOSE_PROMPT.format(family=family,
                              positives=_fmt_pos(positive_cases),
                              negatives=_fmt_neg(negative_cases))
    clause = chat(p, temperature=0.4, max_tokens=400).strip()
    return clause

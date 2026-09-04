# -*- coding: utf-8 -*-
"""Combiner: turn a library of gated narrow clauses (+ optional latent-probe veto)
into a final prediction, and score it. Pure CPU — operates on cached prediction JSONs.

A clause "prediction file" is a list of {id, label, base, new}:
  base = the frozen localizer (v9t) prediction for that case  (-1 = all-correct)
  new  = the clause-augmented prediction
The GATE rule: only use `new` where the base said all-correct (-1); otherwise trust base.
This is the safety invariant that makes adding a clause near-monotone.
"""
import json


def load(path):
    return json.load(open(path, encoding="utf-8"))


def f1(labels, preds):
    eh = et = ch = ct = 0
    for l, p in zip(labels, preds):
        if l == -1:
            ct += 1; ch += (p == -1)
        else:
            et += 1; eh += (p == l)
    ae = eh / max(et, 1); ac = ch / max(ct, 1)
    return round(2 * ae * ac / max(ae + ac, 1e-9) * 100, 1), eh, et, ch, ct


def gated(base, new):
    """Apply one clause in gated mode: fill only the base's all-correct verdicts."""
    return [b if b != -1 else n for b, n in zip(base, new)]


def union(base, clause_news, order=None):
    """Prediction-level union of several gated clauses. For a base-miss case, take the
    first clause (by `order`) that fires (predicts != -1). NEVER merge clauses into one
    prompt — they interfere; union at the prediction level instead."""
    order = order if order is not None else list(range(len(clause_news)))
    out = []
    for i in range(len(base)):
        if base[i] != -1:
            out.append(base[i]); continue
        k = -1
        for j in order:
            v = clause_news[j][i]
            if v != -1:
                k = v; break
        out.append(k)
    return out


def probe_veto(pred, base, probe_scores, topk=2):
    """Latent-probe precision veto. Accept a clause override at step k ONLY if k is among
    the probe's top-`topk` most-suspicious steps for that solution (rank-based ==> scale
    invariant, immune to train->eval score shift). Otherwise revert to all-correct.
    `probe_scores[i]` is a per-step wrongness score list (or None).
    If the probe is entirely unavailable (all None, e.g. no GPU run yet) the veto is
    DISABLED and `pred` is returned unchanged; a per-case None still rejects that case."""
    if probe_scores is None or all(s is None for s in probe_scores):
        return list(pred)
    out = []
    for i in range(len(pred)):
        if base[i] != -1:            # never touch the localizer's own decisions
            out.append(pred[i]); continue
        k = pred[i]
        s = probe_scores[i] if i < len(probe_scores) else None
        if k == -1 or s is None or k >= len(s):
            out.append(-1); continue
        topidx = sorted(range(len(s)), key=lambda j: -s[j])[:topk]
        out.append(k if k in topidx else -1)
    return out


def budget(base, labels, pred):
    """rescue = base-wrong now-correct; broke = base-correct now-wrong."""
    rescue = [i for i in range(len(base)) if base[i] != labels[i] and pred[i] == labels[i]]
    broke = [i for i in range(len(base)) if base[i] == labels[i] and pred[i] != labels[i]]
    return rescue, broke

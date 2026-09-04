# -*- coding: utf-8 -*-
"""Cached Verifier for CPU demos: serves this session's OmniMATH predictions from disk so
loop.py can exercise the gate/union/veto/accept logic without a GPU.

Limits (honest): only the two ALREADY-RUN clauses can be "run" (their outputs are cached);
a genuinely new clause proposed by the Proposer needs the real GPU verifier. cases() returns
full problem/steps only if ProcessBench/omnimath.json is reachable locally, else id+label.
"""
import os, json
from loop import Verifier

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "sample_data")

_KNOWN = {  # family -> cached clause prediction file
    "concrete-number": "omnimath_clause_concrete_number.json",
    "assertion":       "omnimath_clause_assertion.json",
}


class ReplayVerifier(Verifier):
    def __init__(self):
        self._num = json.load(open(os.path.join(D, _KNOWN["concrete-number"]), encoding="utf-8"))
        self._ids = [x["id"] for x in self._num]
        self._label = {x["id"]: x["label"] for x in self._num}
        self._base = {x["id"]: x["base"] for x in self._num}
        # one cached split; dev=val=test here (real runs use disjoint slices)
        pb = os.environ.get("PROCESSBENCH_OMNIMATH", "")
        self._cases = None
        if pb and os.path.exists(pb):
            data = {x["id"]: x for x in json.load(open(pb, encoding="utf-8"))}
            self._cases = [data[i] for i in self._ids if i in data]

    def base_preds(self, split):
        return [self._base[i] for i in self._ids]

    def cases(self, split):
        if self._cases is not None:
            return self._cases
        return [{"id": i, "label": self._label[i], "problem": "", "steps": []} for i in self._ids]

    def run_clause(self, clause, split):
        # cached lookup by family keyword; a new clause would need the GPU verifier
        for fam, f in _KNOWN.items():
            if fam in (clause or "").lower():
                d = json.load(open(os.path.join(D, f), encoding="utf-8"))
                return [x["new"] for x in d]
        raise NotImplementedError("new clause needs the GPU verifier (frozen 7B pass); "
                                  "only cached clauses are replayable on CPU")

    def probe_scores(self, split):
        p = os.path.join(D, "omnimath_probe_scores_L-7.json")
        if os.path.exists(p):
            return json.load(open(p, encoding="utf-8"))
        return [None] * len(self._ids)  # veto becomes a no-op without cached scores

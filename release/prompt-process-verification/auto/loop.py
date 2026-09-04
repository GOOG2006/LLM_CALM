# -*- coding: utf-8 -*-
"""Closed-loop, error-driven prompt optimizer for step-error localization.

    run localizer -> dump wrong (dev) -> tag + cluster checkable families
      -> propose a narrow clause per family -> GATED eval on val
      -> keep clause iff net-positive and no collapse -> grow clause library
      -> combine library (prediction-level union INTERSECT probe-veto) -> report on test

The three safety invariants that make automatic clause-adding safe (and that generic
auto-prompt-opt lacks) are enforced here:
  (1) GATE     : a new clause may only fill the localizer's all-correct verdicts.
  (2) UNION    : clauses are combined at the prediction level, never merged into one prompt.
  (3) PROBE-VETO: a clause override survives only if an in-domain latent probe agrees (top-k).

GPU is needed ONLY inside `verifier.run_clause` (a frozen-7B pass) and probe scoring.
Everything else (tag / propose / combine / accept) is CPU + analyst-LLM calls.
"""
import json, argparse
import combiner as C
from tagger import tag_all, cluster_checkable
from proposer import propose_clause


class Verifier:
    """Abstracts the GPU-side generation. Inject a live one (calls src/gen_localize.py
    over SSH) or a cached one (replay). Must implement:
        base_preds(split)          -> list[int]   (v9t localizer, -1 = all-correct)
        run_clause(clause, split)  -> list[int]   (localizer+clause predictions)
        probe_scores(split)        -> list[list[float]|None]
        cases(split)               -> list[dict]  (id, problem, steps, label)
    """
    def base_preds(self, split): raise NotImplementedError
    def run_clause(self, clause, split): raise NotImplementedError
    def probe_scores(self, split): raise NotImplementedError
    def cases(self, split): raise NotImplementedError


def wrong_cases(cases, base):
    out = []
    for c, p in zip(cases, base):
        if p != c["label"]:
            d = dict(c); d["pred"] = p; out.append(d)
    return out


def optimize(verifier, rounds=3, accept_min_net=1, collapse_cor_drop=3, veto_topk=2, verbose=True):
    dev, val, test = "dev", "val", "test"
    base_val = verifier.base_preds(val)
    lab_val = [c["label"] for c in verifier.cases(val)]
    lab_test = [c["label"] for c in verifier.cases(test)]
    base_test = verifier.base_preds(test)
    library = []                      # list of {family, clause, new_val, new_test}

    for r in range(rounds):
        # 1) analyze localizer failures on dev
        base_dev = verifier.base_preds(dev)
        cases_dev = verifier.cases(dev)
        wrong = wrong_cases(cases_dev, base_dev)
        tags = tag_all(wrong)
        clusters = cluster_checkable(tags)
        if verbose:
            print(f"[round {r}] wrong={len(wrong)} checkable-clusters="
                  f"{[(f, len(x)) for f, x in clusters]}", flush=True)

        # 2) propose + evaluate one clause per top cluster
        neg = [c for c in cases_dev if c["label"] == -1][:3]
        for family, tag_ids in clusters:
            ids = {t["id"] for t in tag_ids}
            pos = [c for c in wrong if c["id"] in ids and c["label"] != -1]
            if not pos:
                continue
            clause = propose_clause(family, pos, neg)
            new_val = verifier.run_clause(clause, val)          # GPU
            gated = C.gated(base_val, new_val)
            rescue, broke = C.budget(base_val, lab_val, gated)
            _, _, _, ch, ct = C.f1(lab_val, gated)
            cor_drop = sum(1 for i in range(len(base_val))
                           if lab_val[i] == -1 and base_val[i] == -1 and gated[i] != -1)
            net = len(rescue) - len(broke)
            keep = net >= accept_min_net and cor_drop <= collapse_cor_drop
            if verbose:
                print(f"  clause[{family}] net={net} (+{len(rescue)}/-{len(broke)}) "
                      f"cor_drop={cor_drop} -> {'KEEP' if keep else 'drop'}", flush=True)
            if keep:
                library.append({"family": family, "clause": clause,
                                "new_val": new_val, "new_test": verifier.run_clause(clause, test)})

        if not library:
            continue
        # 3) combine on val to pick fallback order (num-first style = larger nets first)
        order = sorted(range(len(library)),
                       key=lambda j: -len(C.budget(base_val, lab_val,
                                                   C.gated(base_val, library[j]["new_val"]))[0]))
        union_val = C.union(base_val, [library[j]["new_val"] for j in order])
        pv_val = C.probe_veto(union_val, base_val, verifier.probe_scores(val), topk=veto_topk)
        if verbose:
            print(f"[round {r}] library={len(library)} "
                  f"val: base={C.f1(lab_val, base_val)[0]} union={C.f1(lab_val, union_val)[0]} "
                  f"union+veto={C.f1(lab_val, pv_val)[0]}", flush=True)

    # 4) final report on held-out TEST
    order = sorted(range(len(library)),
                   key=lambda j: -len(C.budget(base_val, lab_val,
                                               C.gated(base_val, library[j]["new_val"]))[0]))
    union_test = C.union(base_test, [library[j]["new_test"] for j in order])
    pv_test = C.probe_veto(union_test, base_test, verifier.probe_scores(test), topk=veto_topk)
    print("\n=== FINAL (held-out test) ===")
    print(f"localizer base   F1={C.f1(lab_test, base_test)[0]}")
    print(f"+clause union    F1={C.f1(lab_test, union_test)[0]}")
    print(f"+union & veto    F1={C.f1(lab_test, pv_test)[0]}")
    print(f"kept clauses: {[l['family'] for l in library]}")
    return library, pv_test


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true", help="CPU-only demo over cached preds")
    args = ap.parse_args()
    if args.replay:
        from replay_verifier import ReplayVerifier
        optimize(ReplayVerifier(), rounds=1)
    else:
        raise SystemExit("Live mode needs the GPU verifier (see README). Use --replay for CPU demo.")

# -*- coding: utf-8 -*-
"""Tagger: read each localizer-wrong case and emit a structured root-cause tag.

The `checkable` flag is the load-bearing output: it separates error families a narrow
clause can catch (arithmetic / enumeration / definition) from CONCEPTUAL proof errors that
no prompt fixes (the frozen-model capability wall). We only spend clause proposals on
checkable clusters.
"""
from llm_backend import chat_json

TAG_PROMPT = """You are auditing a step-level math verifier. For the case below, the
verifier tried to find the FIRST wrong step of a student's solution.

Problem:
{problem}

Numbered steps:
{steps}

Ground-truth first wrong step: {true}   (-1 means the solution is fully correct)
Verifier predicted: {pred}              (-1 means it judged all steps correct)

Classify the verifier's failure. Return ONE JSON object with fields:
- "failure_mode": one of "miss" (true error, verifier said all-correct),
  "late" (verifier flagged a step AFTER the true error), "early" (BEFORE),
  "false_alarm" (solution correct, verifier flagged a step), "other".
- "error_family": short slug for the nature of the TRUE error, e.g. "arithmetic",
  "enumeration" (missed a case / wrong count), "definition" (misread notation/target),
  "algebra", "geometry", "logic-leap", "construction-optimality", "conceptual".
- "checkable": true if a mechanical recheck (recompute a number, search for one more
  case, re-read a definition) could catch this true error; false if it needs genuine
  proof understanding.
- "handle": 3-8 word phrase naming the specific slip (for clustering).
Return only the JSON."""


def tag_case(case):
    steps = "\n".join(f"Step {i+1}: {s.strip()[:300]}" for i, s in enumerate(case["steps"]))
    p = TAG_PROMPT.format(problem=case["problem"][:800], steps=steps,
                          true=(case["label"] + 1 if case["label"] >= 0 else -1),
                          pred=(case["pred"] + 1 if case["pred"] >= 0 else -1))
    t = chat_json(p)
    t["id"] = case["id"]
    return t


def tag_all(wrong_cases):
    return [tag_case(c) for c in wrong_cases]


def cluster_checkable(tags, min_size=2):
    """Group checkable tags by error_family; return clusters sorted by size (desc)."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for t in tags:
        if t.get("checkable") and t.get("failure_mode") in ("miss", "late", "early"):
            buckets[t["error_family"]].append(t)
    clusters = [(fam, ids) for fam, ids in buckets.items() if len(ids) >= min_size]
    clusters.sort(key=lambda x: -len(x[1]))
    return clusters

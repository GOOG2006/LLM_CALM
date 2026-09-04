# -*- coding: utf-8 -*-
"""Pluggable analyst LLM used by the Tagger and Proposer (NOT the frozen verifier).

Two backends:
  * "openai"  : any OpenAI-compatible chat endpoint (env: OPENAI_API_KEY, OPENAI_BASE_URL,
                OPENAI_MODEL). Use a strong model here (analysis quality matters).
  * "local7b" : the same frozen DSR1-7B already on the GPU box, via vLLM. Self-contained,
                no external key, but weaker analysis. (Runs remotely; see src/remote/.)

The analyst runs OFFLINE and does not need to be the verifier. Analysis + clause writing
are the "human" steps we are automating; they are plain text-in/text-out LLM calls.
"""
import os, json


def chat(prompt, system="You are a careful math error analyst.", temperature=0.2, max_tokens=1200):
    backend = os.environ.get("ANALYST_BACKEND", "openai")
    if backend == "openai":
        return _openai(prompt, system, temperature, max_tokens)
    elif backend == "local7b":
        return _local7b(prompt, system, temperature, max_tokens)
    raise ValueError(f"unknown ANALYST_BACKEND={backend}")


def _openai(prompt, system, temperature, max_tokens):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    r = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=temperature, max_tokens=max_tokens)
    return r.choices[0].message.content


def _local7b(prompt, system, temperature, max_tokens):
    # Delegates to a small server-side script over SSH (keeps this file dependency-free).
    # Expects analyst_local.py deployed on the GPU box; returns raw completion text.
    import subprocess, sys
    payload = json.dumps({"system": system, "prompt": prompt,
                          "temperature": temperature, "max_tokens": max_tokens})
    here = os.path.dirname(os.path.abspath(__file__))
    remote = os.path.join(here, "..", "src", "remote", "rrun.py")
    # write payload, push, run analyst_local.py, read back — left as an integration point.
    raise NotImplementedError("local7b backend: deploy analyst_local.py on the GPU box; "
                              "or use ANALYST_BACKEND=openai for the prototype.")


def chat_json(prompt, **kw):
    """Call chat and parse the first JSON object/array in the reply (robust to prose)."""
    txt = chat(prompt, **kw)
    import re
    m = re.search(r"(\{.*\}|\[.*\])", txt, re.S)
    if not m:
        raise ValueError("no JSON in analyst reply:\n" + txt[:500])
    return json.loads(m.group(1))

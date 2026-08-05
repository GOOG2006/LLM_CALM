"""再平衡 CoT+代码 SFT(目标=完整 rationale)。--nofactor 错步过采样。
用法: python train_cot.py --out adapter_cot --nofactor 3 --max 20000 --steps 600"""
import json, argparse, random, torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
                          DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model
BASE = "/root/autodl-tmp/genprm_work/models/DSR1-1.5B"
DATA = "/root/autodl-tmp/genprm_work/sft_cot.jsonl"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--nofactor", type=float, default=3.0)
    ap.add_argument("--max", type=int, default=20000); ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--bs", type=int, default=4); ap.add_argument("--maxlen", type=int, default=1536)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]
    random.seed(0); random.shuffle(rows); rows = rows[: a.max]
    yes = [r for r in rows if r["label"] == "YES"]; no = [r for r in rows if r["label"] == "NO"]
    data = yes + no * int(a.nofactor) + no[: int(len(no) * (a.nofactor % 1))]
    random.shuffle(data)
    print(f"examples={len(data)} (YES={len(yes)} NO_x{a.nofactor}={len(data)-len(yes)})", flush=True)

    def enc(r):
        pids = tok(r["prompt"][:2200], add_special_tokens=True)["input_ids"]
        cids = tok(r["rationale"][-3500:], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        cids = cids[-(a.maxlen - 64):]                      # rationale 保尾(判定在尾)
        keep = a.maxlen - len(cids)
        if len(pids) > keep:
            pids = pids[:1] + pids[-(keep - 1):]
        return {"input_ids": pids + cids, "labels": [-100] * len(pids) + cids}

    ds = Dataset.from_list([enc(r) for r in data])
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16, device_map="cuda")
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.enable_input_require_grads()
    args = TrainingArguments(output_dir=a.out, per_device_train_batch_size=a.bs, gradient_accumulation_steps=4,
        max_steps=a.steps, learning_rate=1e-4, logging_steps=50, save_strategy="no", fp16=True,
        gradient_checkpointing=True, report_to="none", warmup_steps=30)
    Trainer(model=model, args=args, train_dataset=ds,
            data_collator=DataCollatorForSeq2Seq(tok, label_pad_token_id=-100, padding=True)).train()
    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    print("SAVED ->", a.out, flush=True)

if __name__ == "__main__":
    main()

"""rigorous 去相关 SFT: 固定50/50类平衡,--bwfactor 控类内"base判错"样本上采样倍数(1=纯平衡A'; >1=去相关B)。
用法: python train_judge3.py --out adapter_A --bwfactor 1 --percls 8000 --steps 700"""
import json, argparse, random, torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model

BASE = "/root/autodl-tmp/genprm_work/models/DSR1-1.5B"
BV = "/root/autodl-tmp/genprm_work/base_verdict.json"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--bwfactor", type=int, default=1)
    ap.add_argument("--percls", type=int, default=8000); ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--bs", type=int, default=8); ap.add_argument("--maxlen", type=int, default=1024)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    data = json.load(open(BV))
    random.seed(0)

    def build_class(lab):
        pool = [r for r in data if r["label"] == lab]
        weighted = []
        for r in pool:
            weighted += [r] * (a.bwfactor if r["base_wrong"] else 1)  # 类内: base判错的上采样
        return [random.choice(weighted) for _ in range(a.percls)]     # 采到固定 percls(50/50)

    rows = build_class("YES") + build_class("NO"); random.shuffle(rows)
    bw = sum(r["base_wrong"] for r in rows)
    print(f"examples={len(rows)} (50/50) base判错占比={100*bw/len(rows):.1f}% bwfactor={a.bwfactor}", flush=True)

    def enc(r):
        pids = tok(r["prompt"] + "\nJudgement:", add_special_tokens=True)["input_ids"]
        cids = tok(" " + r["label"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        keep = a.maxlen - len(cids)
        if len(pids) > keep:
            pids = pids[:1] + pids[-(keep - 1):]
        return {"input_ids": pids + cids, "labels": [-100] * len(pids) + cids}

    ds = Dataset.from_list([enc(r) for r in rows])
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16, device_map="cuda")
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.enable_input_require_grads()
    args = TrainingArguments(output_dir=a.out, per_device_train_batch_size=a.bs, gradient_accumulation_steps=2,
        max_steps=a.steps, learning_rate=1e-4, logging_steps=50, save_strategy="no", fp16=True,
        gradient_checkpointing=True, report_to="none", warmup_steps=30)
    Trainer(model=model, args=args, train_dataset=ds,
            data_collator=DataCollatorForSeq2Seq(tok, label_pad_token_id=-100, padding=True)).train()
    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    print("SAVED ->", a.out, flush=True)

if __name__ == "__main__":
    main()

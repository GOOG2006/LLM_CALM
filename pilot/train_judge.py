"""判定-only LoRA SFT。--nofactor 控错步(NO)过采样倍数(1=均匀标准SFT; >1=去相关/错步加权)。
用法: python train_judge.py --out adapter_std --nofactor 1 --max 12000 --steps 400"""
import os, json, argparse, random, torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

BASE = "/root/autodl-tmp/genprm_work/models/DSR1-1.5B"
DATA = "/root/autodl-tmp/genprm_work/sft_judg.jsonl"
RESP = "\nJudgement:"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--nofactor", type=float, default=1.0)   # 错步过采样倍数
    ap.add_argument("--max", type=int, default=12000)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]
    random.seed(0); random.shuffle(rows); rows = rows[: a.max]
    # 错步过采样
    yes = [r for r in rows if r["label"] == "YES"]; no = [r for r in rows if r["label"] == "NO"]
    no_os = no * int(a.nofactor) + no[: int(len(no) * (a.nofactor % 1))]
    data = yes + no_os; random.shuffle(data)
    texts = [{"text": r["prompt"][:2600] + RESP + " " + r["label"]} for r in data]
    print(f"train examples={len(texts)} (YES={len(yes)} NO_orig={len(no)} NO_after={len(no_os)}) nofactor={a.nofactor}", flush=True)
    ds = Dataset.from_list(texts)

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map="cuda")
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    cfg = SFTConfig(output_dir=a.out, per_device_train_batch_size=a.bs, gradient_accumulation_steps=2,
                    max_steps=a.steps, learning_rate=1e-4, logging_steps=25, save_strategy="no",
                    fp16=True, gradient_checkpointing=True, report_to="none",
                    dataset_text_field="text")
    tr = SFTTrainer(model=model, train_dataset=ds, args=cfg, peft_config=lora)
    tr.train()
    tr.model.save_pretrained(a.out); tok.save_pretrained(a.out)
    print("SAVED adapter ->", a.out, flush=True)

if __name__ == "__main__":
    main()

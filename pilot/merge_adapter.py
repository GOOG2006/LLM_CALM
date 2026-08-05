"""合并 LoRA adapter 进 base,存完整模型 + 拷可用 tokenizer。用法: python merge_adapter.py adapter_std merged_std"""
import sys, shutil, os, torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
BASE = "/root/autodl-tmp/genprm_work/models/DSR1-1.5B"
GOODTOK = "/root/autodl-tmp/genprm_work/models/GenPRM-1.5B"
adapter, out = sys.argv[1], sys.argv[2]
base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16)
m = PeftModel.from_pretrained(base, os.path.join("/root/autodl-tmp/genprm_work", adapter))
m = m.merge_and_unload()
m.save_pretrained(os.path.join("/root/autodl-tmp/genprm_work", out))
for fn in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]:
    src = os.path.join(GOODTOK, fn)
    if os.path.exists(src):
        shutil.copy(src, os.path.join("/root/autodl-tmp/genprm_work", out, fn))
print("MERGED ->", out)

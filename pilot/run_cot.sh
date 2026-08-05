cd /root/autodl-tmp/genprm_work
PY=envs/GenPRM/bin/python
echo "=== train CoT-SFT rebalanced (nofactor=3) ==="; $PY train_cot.py --out adapter_cot --nofactor 3 --max 20000 --steps 600 --bs 4
echo "=== merge CoT ==="; $PY merge_adapter.py adapter_cot merged_cot
echo "=== EVAL CoT ==="; $PY eval_cot.py --model merged_cot --config gsm8k --n 100
echo "=== COT DONE ==="

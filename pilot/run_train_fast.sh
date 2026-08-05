cd /root/autodl-tmp/genprm_work
PY=envs/GenPRM/bin/python
echo "=== [1] train standard SFT (nofactor=1) ==="; $PY train_judge2.py --out adapter_std2 --nofactor 1 --max 24000 --steps 700 --bs 8
echo "=== [2] merge std ==="; $PY merge_adapter.py adapter_std2 merged_std2
echo "=== [3] EVAL standard ==="; $PY eval_hf.py --model merged_std2 --config gsm8k --n 100
echo "=== [4] train decorr-weighted SFT (nofactor=3) ==="; $PY train_judge2.py --out adapter_w2 --nofactor 3 --max 24000 --steps 700 --bs 8
echo "=== [5] merge weighted ==="; $PY merge_adapter.py adapter_w2 merged_w2
echo "=== [6] EVAL weighted ==="; $PY eval_hf.py --model merged_w2 --config gsm8k --n 100
echo "=== ALL DONE ==="

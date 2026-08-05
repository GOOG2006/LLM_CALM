cd /root/autodl-tmp/genprm_work
PY=envs/GenPRM/bin/python
echo "=== [1/6] train standard SFT (nofactor=1) ==="; $PY train_judge2.py --out adapter_std2 --nofactor 1 --max 30000 --steps 1500 --bs 8
echo "=== [2/6] train decorr-weighted SFT (nofactor=3) ==="; $PY train_judge2.py --out adapter_w2 --nofactor 3 --max 30000 --steps 1500 --bs 8
echo "=== [3/6] merge std ==="; $PY merge_adapter.py adapter_std2 merged_std2
echo "=== [4/6] merge weighted ==="; $PY merge_adapter.py adapter_w2 merged_w2
echo "=== [5/6] eval standard ==="; $PY eval_hf.py --model merged_std2 --config gsm8k --n 120
echo "=== [6/6] eval weighted ==="; $PY eval_hf.py --model merged_w2 --config gsm8k --n 120
echo "=== ALL DONE ==="

cd /root/autodl-tmp/genprm_work
PY=envs/GenPRM/bin/python
while [ ! -f base_verdict.json ]; do sleep 15; done
sleep 5
echo "=== train A' (balance only, bwfactor=1) ==="; $PY train_judge3.py --out adapter_A --bwfactor 1 --percls 8000 --steps 700
echo "=== merge A' ==="; $PY merge_adapter.py adapter_A merged_A
echo "=== EVAL A' (rebalance-only) ==="; $PY eval_hf.py --model merged_A --config gsm8k --n 100
echo "=== train B (decorr, bwfactor=3) ==="; $PY train_judge3.py --out adapter_B --bwfactor 3 --percls 8000 --steps 700
echo "=== merge B ==="; $PY merge_adapter.py adapter_B merged_B
echo "=== EVAL B (decorrelation) ==="; $PY eval_hf.py --model merged_B --config gsm8k --n 100
echo "=== RIGOROUS DONE ==="

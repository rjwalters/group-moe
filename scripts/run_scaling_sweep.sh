#!/bin/bash
# S_n scaling sweep: n=3,4,5 × 3 seeds × 3 models
# Expected runtime: ~2-4 hours on MPS

cd /Users/rwalters/GitHub/group-moe

echo "=== S_n Scaling Sweep ==="
echo "Started: $(date)"
echo ""

for n in 3 4 5; do
    for seed in 42 123 7; do
        outdir="data/scaling_sn/s${n}_seed${seed}"
        echo "--- S_${n} seed=${seed} ---"
        uv run python scripts/train_nary.py \
            --n $n --model all --epochs 500 --patience 75 \
            --num-range 8 --d-model 128 --n-blocks 2 \
            --balance-alpha 0.01 --log-every 9999 \
            --seed $seed --output-dir $outdir 2>&1 | grep -E "(COMPARISON|comp=)"
        echo ""
    done
done

echo "=== Sweep Complete ==="
echo "Finished: $(date)"

# Summarize
echo ""
echo "=== FULL SUMMARY ==="
for n in 3 4 5; do
    echo "S_${n}:"
    for seed in 42 123 7; do
        f="data/scaling_sn/s${n}_seed${seed}/results.json"
        if [ -f "$f" ]; then
            python3 -c "
import json
d = json.load(open('$f'))
for m in ['groupmoe', 'standardmoe', 'baseline']:
    if m in d:
        acc = d[m]['test'][-1].get('complement_acc', 0)
        print(f'  seed=$seed {m:14s}: {acc:.4f}')
"
        fi
    done
done

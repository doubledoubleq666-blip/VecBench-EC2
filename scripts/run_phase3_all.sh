#!/bin/bash
# Phase 3: Run all benchmarks sequentially
# Usage: nohup bash scripts/run_phase3_all.sh > results/phase3_full.log 2>&1 &
set -e
cd ~/vector-db-bench
source venv/bin/activate

SCALES=("10K:data/processed/sift_10k.npy" "100K:data/processed/sift_100k.npy" "1M:data/processed/sift_1m.npy")
THREADS="1,4,8,16"
REPEAT=3

run_db() {
    local db=$1
    local compose_dir=$2

    echo "============================================"
    echo "Starting $db at $(date)"
    echo "============================================"

    # Start database
    cd ~/vector-db-bench/docker/$compose_dir
    docker compose up -d
    echo "Waiting for $db to be ready..."
    sleep 30
    cd ~/vector-db-bench

    for entry in "${SCALES[@]}"; do
        IFS=':' read -r scale dataset <<< "$entry"
        echo ""
        echo ">>> $db | $scale | $(date)"
        python benchmark.py \
            --db "$db" \
            --dataset "$dataset" \
            --scale "$scale" \
            --threads "$THREADS" \
            --repeat "$REPEAT" \
            --num-queries 100 \
            --output results \
            || echo "WARN: $db $scale failed, continuing..."
    done

    # Stop database
    cd ~/vector-db-bench/docker/$compose_dir
    docker compose down
    cd ~/vector-db-bench

    echo "$db completed at $(date)"
    echo ""
}

echo "Phase 3 started at $(date)"
echo ""

# Run each database
run_db "milvus" "milvus"
run_db "chroma" "chroma"
run_db "weaviate" "weaviate"

echo ""
echo "Phase 3 completed at $(date)"
echo "Results:"
ls -lh results/*.csv

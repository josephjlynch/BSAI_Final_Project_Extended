#!/bin/bash
# Wrapper: keeps re-running extract_spike_times.py until NPZ count stops growing.
# Each Python run extracts 1-2 sessions before macOS kills it; this restarts it.

PYTHON="/Users/donespejo/Documents/TsinghuaFall2025/Machine Learning/Conversations for BSAI/BSAI_Final_Project_Extended/venv/bin/python"
DIR="/Users/donespejo/Documents/TsinghuaFall2025/Machine Learning/BSAI_Final_Project_Extended"
NPZ_DIR="$DIR/results/derivatives/spike_times"
LOG="/tmp/extraction_wrapper.log"
MAX_RUNS=20

cd "$DIR"

prev_count=-1
for i in $(seq 1 $MAX_RUNS); do
    count=$(ls "$NPZ_DIR" | wc -l | tr -d ' ')
    echo "[$(date +%H:%M:%S)] Run $i — NPZ before: $count" | tee -a "$LOG"

    if [ "$count" -eq "$prev_count" ]; then
        echo "[$(date +%H:%M:%S)] No new sessions extracted. All valid sessions done." | tee -a "$LOG"
        break
    fi
    prev_count=$count

    "$PYTHON" extract_spike_times.py >> "$LOG" 2>&1
    new_count=$(ls "$NPZ_DIR" | wc -l | tr -d ' ')
    echo "[$(date +%H:%M:%S)] Run $i done — NPZ after: $new_count" | tee -a "$LOG"
    sleep 3
done

echo "" | tee -a "$LOG"
echo "=== FINAL NPZ COUNT: $(ls "$NPZ_DIR" | wc -l | tr -d ' ') ===" | tee -a "$LOG"

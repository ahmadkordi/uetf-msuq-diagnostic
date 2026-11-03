#!/bin/bash
# Runs 12-cell correctness scoring in 4 parallel streams, one per Ollama port.

SRC=/mnt/src/score_correctness.py
OUT=/mnt/output
MODEL=qwen2.5:32b
WORKERS=8

run_stream () {
  local port=$1
  shift
  local cells=("$@")
  export OLLAMA_HOST="http://localhost:${port}"
  for cell in "${cells[@]}"; do
    infile="stage_a_${cell}.jsonl"
    echo "[port $port] === $cell ==="
    python "$SRC" \
      --input "$OUT/$infile" \
      --output "$OUT/stage_h_correctness_${cell}.jsonl" \
      --model "$MODEL" \
      --workers "$WORKERS"
  done
  echo "[port $port] STREAM DONE"
}

# Stream 1 - port 11434 (GPU 0)
run_stream 11434 \
  coqa_Llama2_13b \
  coqa_Q2.5_14b \
  coqa_llama3.1_8b &

# Stream 2 - port 11435 (GPU 3)
run_stream 11435 \
  hotpotqa_Llama2_13b \
  hotpotqa_Q2.5_14b \
  hotpotqa_llama3.1_8b &

# Stream 3 - port 11436 (GPU 1)
run_stream 11436 \
  nqopen_Llama2_13b \
  nqopen_Q2.5_14b \
  nqopen_llama3.1_8b &

# Stream 4 - port 11437 (GPU 2)
run_stream 11437 \
  truthfulqa_Llama2_13b \
  truthfulqa_Q2.5_14b \
  truthfulqa_llama3.1_8b &

wait
echo "ALL 12 CELLS DONE"

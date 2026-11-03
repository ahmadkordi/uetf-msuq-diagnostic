#!/usr/bin/env bash
# Experiment 0 (0B, 0C, 0D) at the primary threshold tau = 70 for the 12 main cells.
# CPU only. Train+val queries only (never test). Reads stored files; writes only new files:
#   $O/experiment_0b/tau70/*                       (0B decomposition at tau = 70, tables)
#   $O/experiment_0b/exp0cd_thr70_cal_12cells.*    (0C/0D at tau = 70)
# Run from the folder with the pipeline scripts (/mnt/src), venv active:
#   bash run_exp0_tau70.sh check     # read-only: inputs present? earlier tau = 70 file? targets free?
#   bash run_exp0_tau70.sh decomp    # 0B decomposition at tau = 70 (+ check that tau = 90 reproduces Table 3)
#   bash run_exp0_tau70.sh cd        # 0C and 0D at tau = 70, 1000 permutations
#   bash run_exp0_tau70.sh tables    # tau = 70 Tables 3 and 4, 0C, comparison with tau = 90 -> tau70/exp0_tau70_tables.md
#   bash run_exp0_tau70.sh all       # decomp, cd, tables
O=${O:-/mnt/output}; E=$O/experiment_0b; S=$O/experiment_0a/splits; L=${L:-/mnt/data/tmp/exp0_tau70}
mkdir -p "$L"
C="coqa_Llama2_13b hotpotqa_Llama2_13b nqopen_Llama2_13b truthfulqa_Llama2_13b \
coqa_llama3.1_8b hotpotqa_llama3.1_8b nqopen_llama3.1_8b truthfulqa_llama3.1_8b \
coqa_Q2.5_14b hotpotqa_Q2.5_14b nqopen_Q2.5_14b truthfulqa_Q2.5_14b"
TAG=thr70_cal_12cells

do_check() {
  local miss=0
  for c in $C; do
    for f in "$E/ssl_cal_$c.jsonl" "$E/stage_g_trainfit_$c.jsonl" "$O/stage_h_correctness_$c.jsonl" \
             "$S/$c.train_ids.txt" "$S/$c.val_ids.txt" "$E/exp0b_decomp_cal_$c.json"; do
      [ -s "$f" ] || { echo "MISSING $f"; miss=1; }
    done
  done
  [ $miss = 0 ] && echo "all inputs present for 12 cells"
  for f in exp0cd_thr90_cal_v2.txt exp0cd_thr70.txt; do
    [ -s "$E/$f" ] && echo "$f: cells $(grep -c '^== \|^\s*== ' "$E/$f")" || echo "$f: not found"
  done
  [ -e "$E/exp0cd_$TAG.txt" ] && echo "TARGET EXISTS: $E/exp0cd_$TAG.txt" || echo "target exp0cd_$TAG.* is free"
  [ -d "$E/tau70" ] && { echo "folder $E/tau70 exists:"; ls "$E/tau70" | head; } || echo "folder $E/tau70 is free"
  python -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"
  for p in exp0b_decompose.py exp0cd_oracle.py exp0_tau70.py exp0_tau70_tables.py; do [ -s "$p" ] || echo "MISSING script $p (run from the code folder)"; done
}
do_decomp() { python exp0_tau70.py --data-dir "$O" --check90 2>&1 | tee "$L/decomp.log"; }
do_cd() {
  [ -e "$E/exp0cd_$TAG.txt" ] && { echo "$E/exp0cd_$TAG.txt exists; stop"; return 1; }
  python exp0cd_oracle.py --data-dir "$O" --cells $C --threshold 70 --ssl-variant cal --n-perm 1000 --tag $TAG 2>&1 \
    | tee "$L/cd.log" | grep "==\|0C entropy\|0D"
}
do_tables() { python exp0_tau70_tables.py --data-dir "$O" --cd70 "exp0cd_$TAG.txt" 2>&1 | tee "$L/tables.log"; }

case "${1:-}" in
  check)  do_check ;;
  decomp) do_decomp ;;
  cd)     do_cd ;;
  tables) do_tables ;;
  all)    do_decomp && do_cd && do_tables ;;
  *) echo "usage: bash run_exp0_tau70.sh check|decomp|cd|tables|all"; exit 1 ;;
esac

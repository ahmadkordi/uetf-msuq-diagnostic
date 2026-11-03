#!/bin/bash
# usage: bash run_post.sh <dataset> <base_cell>   e.g. truthfulqa truthfulqa_Llama2_13b
ds=$1; b=$2; c=${ds}_Llama2chat_13b; O=/mnt/output; S=$O/experiment_0a/splits; E=$O/experiment_0b; PAR=${PAR:-8}
source ~/venvs/uncertainty/bin/activate; cd /mnt/src; mkdir -p $E/bootstrap_shards
while [ ! -s $O/stage_f_estimators_$c.jsonl ]; do echo "$(date +%T) waiting for stage_f"; sleep 120; done
for s in train val test; do [ -s $S/$c.${s}_ids.txt ] || cp $S/$b.${s}_ids.txt $S/$c.${s}_ids.txt; done
[ -s $E/stage_g_trainfit_$c.jsonl ] || python calibrate_ecdf.py --stage-f $O/stage_f_estimators_$c.jsonl --tensor $O/stage_c_tensor_$c.jsonl --train-ids $S/$c.train_ids.txt --output $E/stage_g_trainfit_$c.jsonl || exit 1
n=$(grep -c . $O/stage_b_s1_semantic_$c.jsonl); shards=$(( (n + 399) / 400 ))
for ((k=0;k<shards;k++)); do
  o=$E/bootstrap_shards/$c.shard${k}of${shards}.jsonl
  [ -s $o ] || { python bootstrap_stability.py --s1 $O/stage_b_s1_semantic_$c.jsonl --s2 $O/stage_b_s2_claim_$c.jsonl --s3 $O/stage_b_s3_self_eval_$c.jsonl --output $o.part --shard-index $k --num-shards $shards && mv $o.part $o; } &
  while [ $(jobs -r | wc -l) -ge $PAR ]; do sleep 5; done
done; wait
python aggregate_bootstrap.py --cells $c || exit 1
[ -s $E/ssl_cal_$c.jsonl ] || python ssl_signal.py --calibrated $E/stage_g_trainfit_$c.jsonl --bootstrap $E/bootstrap_sk_cal_$c.jsonl --train-ids $S/$c.train_ids.txt --output $E/ssl_cal_$c.jsonl || exit 1
echo "$ds post DONE"

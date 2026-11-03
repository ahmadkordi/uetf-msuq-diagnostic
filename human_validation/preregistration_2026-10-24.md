# Pre-specified analysis plan: human validation of the correctness judge

Recorded on 2026-10-24, after the pilot round and before the main sample was drawn.

1. **Primary statistic.** Cohen's kappa between the judge label and the final human label,
   averaged over the four datasets (macro), weighted to the population of judged
   responses within each dataset; 95% stratified bootstrap interval, 2,000 resamples.
2. **Threshold rule.** If the 95% interval of the kappa difference (90 minus 70) excludes 0,
   the threshold with the higher kappa is named the primary threshold in the paper.
   Otherwise both thresholds are reported side by side, as now.
3. **UNCLEAR.** Excluded from the primary analysis; counted as INCORRECT in a sensitivity
   analysis.
4. **Annotator agreement floor.** Binary kappa between the two annotators of at least 0.6.
   Below that, the guidelines are revised and a new sample is drawn with a new seed and
   output folder; the same items are not re-labelled.
5. **Judge agreement floor.** If judge-human kappa is below 0.6 at both thresholds, the
   paper calls the labels "judge-assessed correctness" and states this in the Limitations.

Main sample: config_uetf_main.yaml, seed 20261020, 4 datasets x 3 judge-score bands,
generators Llama2_13b, Llama2chat_13b, llama3.1_8b, Q2.5_14b;
48 responses per dataset x band (12 per generator), 576 responses in total. Guidelines: version 1.0 with Section 10 (D1-D10).
After sampling, record the sha256 of KEY_do_not_open.csv from sampling_meta.json here:

key_sha256 = 1822145804f541f6b2ab7fec0c1ef24d66b3ffe58d33ae65e3741ffbc21fee18

Note: a first draw with three generators (480 responses) was replaced by this draw before
any item was labelled, to add the Llama-2-13B chat cells.

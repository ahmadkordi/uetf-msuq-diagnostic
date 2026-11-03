# uetf-msuq-diagnostic

Code and human labels for a negative-result, diagnostic study of **MS-UQ**-style
uncertainty estimation: a query-conditioned, self-supervised reliability gate over
K = 15 black-box LLM uncertainty estimators. The study asks whether such a gate beats
simple alternatives (for example the plain mean of the estimator pool) and reports
that, in the specification evaluated, it does not.

This repository accompanies a manuscript under preparation. A link to the paper will
be added here.

## What is in this repository

| Path | Content |
|---|---|
| `code/` | The pipeline scripts, one stand-alone script per stage, with fixed seeds |
| `code/tools/` | Helper scripts used while running the experiments (progress monitors, read-only checks); not needed to reproduce the tables |
| `human_validation/final_labels.csv` | Human validation of the correctness judge: 576 stratified responses |

Not included: model weights, generated responses, intermediate stage outputs and the
benchmark data. These are large and are obtained from their original sources (see
"Data and models").

## Setup of the study

- **Datasets:** CoQA, HotpotQA, NQ-Open, TruthfulQA.
- **Generators:** Llama-2-13B (base), Llama-3.1-8B-Instruct, Qwen2.5-14B-Instruct
  (12 main cells), plus Llama-2-13B-chat (4 separate chat cells).
- **Sampling:** 20 responses per query, temperature 1.0, top-p 1.0, top-k off,
  256 new tokens, seed 20260903, bf16, no 4-bit quantisation.
- **Correctness judge:** Qwen2.5-32B (`qwen2.5:32b`, Q4_K_M) through Ollama,
  temperature 0, correct if score above 70 (main threshold; 90 as a sensitivity analysis).
- **Claim extraction:** `qwen:32b` (Q4_0) through Ollama.
- **NLI:** `deberta-large-mnli`.
- **Split:** hash-based train/validation/test split, seed 20260903 (CoQA grouped by conversation).

## Pipeline

The scripts correspond to the stages of the paper (Appendix G lists them in full).
Roughly, in order:

1. **Generation:** `generate_responses*.py` (per dataset; `*_shard.py` for sharded runs); chat cells: `run_stage_a_chat.py`.
2. **Claim extraction and NLI:** `extract_claims.py`, `semantic_nli.py`, `claim_nli.py`.
3. **Self-evaluation (P(true)):** `self_eval.py` (`self_eval_shard.py`, `self_eval_share.py` for sharded runs).
4. **Graph and tensor estimators:** `graph_estimators.py`, `tensor_decompose.py`; normalisation: `calibrate_ecdf.py`.
5. **Correctness scoring:** `score_correctness.py`.
6. **Audits and stability:** `dependency_audit.py`, `bootstrap_stability.py`, `run_bootstrap_sharded.py`, `aggregate_bootstrap.py`.
7. **Gate, baselines and tables:** `ssl_signal.py`, `gate_eval.py`, `baseline_auroc.py`, `supervised_ceiling.py`, `summarize_gate.py`, `stats_extra.py`.
8. **Additional baselines and diagnostics:** `extra_baselines.py`, `kle_baseline.py`, `estimator_vs_b3.py`, `experiment_0b.py`, `exp0cd_oracle.py`, `tensor_sem_only.py`, `nosym_eval.py`.
9. **Base versus chat replication:** `compare_base_chat.py`, `exp0_compare_base_chat.py`, `base_chat_causes.py`.

Scripts named `diagnose_*`, `ptrue_*`, `token_test.py`, `check_restore.py` and
`run_*.sh` are diagnostic checks or drivers written during the study; they are not
required to reproduce the main tables.

## Paths

The scripts were run with the layout `/mnt/src` (code), `/mnt/output` (stage outputs),
`/mnt/data/models` (model checkpoints) and `/mnt/data/tmp`. These are the paths of
our server. They must be adapted to your environment before running.

## Human validation labels

`human_validation/final_labels.csv` has one row per validated response (576 rows).

| Column | Meaning |
|---|---|
| `item_id` | Anonymous item identifier |
| `dataset`, `model`, `query_id`, `sample_idx` | Cell and response identifiers |
| `judge_score` | Score given by the LLM judge (0 to 100) |
| `band_label` | Judge band used for stratification (`le70`, `70to90`, `gt90`) |
| `label_A`, `label_B` | Independent labels of the two annotators (`CORRECT`, `INCORRECT`, `UNCLEAR`) |
| `final_label` | Final label: the agreed label, or the adjudicated label where the annotators disagreed |
| `label_source` | `both_agree` or `adjudicated` |

The sample has 48 strata (dataset by generator by judge band), 12 responses each.
Seven responses remain `UNCLEAR` after adjudication and are excluded from the main analysis.
Question and response texts are not included; they can be regenerated from the benchmarks
and the pipeline outputs using `dataset`, `query_id` and `sample_idx`.

## Data and models

Benchmarks (CoQA, HotpotQA, NQ-Open, TruthfulQA) and model checkpoints are used under
their own licences and are not redistributed here. Checkpoints used:
`Llama-2-13b-hf`, `Llama-2-13b-chat-hf`, `Meta-Llama-3.1-8B-Instruct`,
`Qwen2.5-14B-Instruct`, `Qwen2.5-32B` (judge), `deberta-large-mnli`.

## Known limitations of this release

- The annotation sampling and analysis scripts and some stage drivers are not yet in this repository.
- The scripts are research code, not a packaged library, and expect the directory layout above.
- Our scripts follow the released MS-UQ protocol where possible; every deviation is documented in the paper (Table 9).

## Licence

See `LICENSE`.

## Citation

To be added.

# Human validation of the correctness judge

This folder holds everything needed to inspect the human validation of the LLM correctness
judge (paper, Section 4.6 and Appendix K): the labels, the sampling record, the kit that
drew and analysed the sample, the guidelines and the pre-specified analysis plan.

## Contents

| Path | Content |
|---|---|
| `final_labels.csv` | 576 validated responses: both annotators' labels, the final label, the judge score |
| `annotation_guidelines_en.md`, `annotation_guidelines_fa.md` | Guidelines given to the annotators (English is the reference; Persian is a translation) |
| `preregistration_2026-10-24.md` | Pre-specified analysis plan and the SHA-256 of the sampling key |
| `sampling/` | `sampling_meta.json`, `sampling_report.md`, `population_counts.csv` of the main draw |
| `kit/` | `sample_for_annotation.py`, `analyze_annotations.py`, `annot_common.py`, `config.example.yaml`, `config_uetf_main.yaml`, `config_uetf_pilot.yaml` |

## Sample

- 576 responses, 4 datasets x 4 generators x 3 judge-score bands (at most 70, 70 to 90, above 90),
  12 per cell, 48 per dataset and band; seed 20261020.
- Generators: Llama-2-13B (base), Llama-2-13B (chat), Llama-3.1-8B, Qwen2.5-14B.
- Two annotators labelled every response independently and blind to the generator and the
  judge score. Disagreements (39) were adjudicated jointly before unblinding.
- A first draw with three generators (480 responses) was replaced by this draw before any
  item was labelled.

## Reproducing the sampling record

`config_uetf_main.yaml` is the configuration used for the main draw. Its SHA-256 is
`76b5cdf88072a98c21c17081c22152a568caa295a4582a2822a32adfc5b16753`, the value recorded as
`config_sha256` in `sampling/sampling_meta.json`. The sampling key (`KEY_do_not_open.csv`, the
map from item id to generator and judge score) is not released; its SHA-256 is recorded as
`key_sha256` (`1822145804f541f6b2ab7fec0c1ef24d66b3ffe58d33ae65e3741ffbc21fee18`) in the
sampling record and in the analysis plan. The same information is in `final_labels.csv`.

The paths in the configuration files (`/mnt/output/...`) are those of our server and must be
adapted. Running the sampling again needs the stage outputs of the pipeline, which are not
part of this repository.

## Kit usage

```
python kit/sample_for_annotation.py --config kit/config.example.yaml --inspect   # look at your files
python kit/sample_for_annotation.py --config kit/config.example.yaml --dry-run   # population counts
python kit/sample_for_annotation.py --config kit/config.example.yaml             # write the sheets
python kit/analyze_annotations.py --dir <output_dir> --pilot                     # after the pilot
python kit/analyze_annotations.py --dir <output_dir>                             # agreement, adjudication sheet, then results
```

Requirements: pandas, numpy, pyyaml, openpyxl (optional; without it the sheets are CSV),
matplotlib (optional, for the threshold-sweep figure). The analysis script writes its own
`results/final_labels.csv` with its own column names; the file in this folder was assembled
from the two annotator sheets and the adjudication sheet and carries the columns listed below.

## Columns of `final_labels.csv`

| Column | Meaning |
|---|---|
| `item_id` | Anonymous item identifier |
| `dataset`, `model`, `query_id`, `sample_idx` | Cell and response identifiers |
| `judge_score` | Score of the LLM judge (0 to 100) |
| `band_label` | Judge-score band used for stratification (`le70`, `70to90`, `gt90`) |
| `label_A`, `label_B` | Independent labels of the two annotators |
| `final_label` | Agreed label, or the adjudicated label where the annotators differed |
| `label_source` | `both_agree` or `adjudicated` |

Question and response texts are not included.

## Provenance notes

- **Guidelines.** The guidelines file available on our server was last modified on
  2026-10-20 and had Section 10 empty. Decisions D1 to D10 in Section 10 are taken from the
  summary in Appendix K.2 of the paper; the original frozen file was not found.
- **Analysis plan date.** The date in `preregistration_2026-10-24.md` is the one recorded in
  the file itself. The commit date of this repository is later. The key hash line was added
  after the draw, as the file states.
- **Annotators.** One annotator is the first author; the other is Samira Nazari. The first
  author re-read his own labels before the judge scores were unblinded and changed 6 of them
  (all towards INCORRECT); see Appendix K of the paper.

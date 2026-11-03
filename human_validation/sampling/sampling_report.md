# Sampling report

- created: 2026-10-24T18:57:46+00:00
- seed: 20261020; thresholds: [70.0, 90.0]; correct if score > threshold
- population: 1585200 responses (0 dropped for a missing score)
- main sample: 576 responses; pilot: 0
- double annotation fraction: 1
- main items sharing a group with another item: 0
- sha256 of KEY_do_not_open.csv: `1822145804f541f6b2ab7fec0c1ef24d66b3ffe58d33ae65e3741ffbc21fee18`

## Main sample per dataset x band x model

```
model                  Llama2_13b  Llama2chat_13b  Q2.5_14b  llama3.1_8b  total
dataset    band_label                                                          
coqa       70to90              12              12        12           12     48
           gt90                12              12        12           12     48
           le70                12              12        12           12     48
hotpotqa   70to90              12              12        12           12     48
           gt90                12              12        12           12     48
           le70                12              12        12           12     48
nqopen     70to90              12              12        12           12     48
           gt90                12              12        12           12     48
           le70                12              12        12           12     48
truthfulqa 70to90              12              12        12           12     48
           gt90                12              12        12           12     48
           le70                12              12        12           12     48
```

## Population (responses) per dataset x model x band

```
band_label                  le70  70to90   gt90
dataset    model                               
coqa       Llama2_13b      32532   64642  62486
           Llama2chat_13b  15267   65207  79186
           Q2.5_14b         6351   75243  78066
           llama3.1_8b     17425   68093  74142
hotpotqa   Llama2_13b      71270   40112  36718
           Llama2chat_13b  30117   76466  41517
           Q2.5_14b        69651   54794  23655
           llama3.1_8b     50577   54231  43292
nqopen     Llama2_13b      41070   21884   9246
           Llama2chat_13b  32692   25763  13745
           Q2.5_14b        36284   23131  12785
           llama3.1_8b     42342   19783  10075
truthfulqa Llama2_13b      12223    3747    370
           Llama2chat_13b   5954    8454   1932
           Q2.5_14b         7536    7748   1056
           llama3.1_8b     10786    4988    566
```

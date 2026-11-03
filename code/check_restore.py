#!/usr/bin/env python3
"""Check that the restored stage_h_correctness files for the four Llama-2-13B cells reproduce B2 and B3-Sum (test AUROC) of the paper (4 digits).
Run from /mnt/src: python check_restore.py"""
import numpy as np
from pathlib import Path
from gate_eval import Cell
from baseline_auroc import official_auroc
REF = {('coqa_Llama2_13b', 'B2'): (0.7584, 0.7792), ('coqa_Llama2_13b', 'B3-Sum'): (0.7919, 0.7534), ('hotpotqa_Llama2_13b', 'B2'): (0.708, 0.7001), ('hotpotqa_Llama2_13b', 'B3-Sum'): (0.7429, 0.7061), ('nqopen_Llama2_13b', 'B2'): (0.6937, 0.6959), ('nqopen_Llama2_13b', 'B3-Sum'): (0.7965, 0.7091), ('truthfulqa_Llama2_13b', 'B2'): (0.8022, 0.6445), ('truthfulqa_Llama2_13b', 'B3-Sum'): (0.8779, 0.5795)}
O = Path("/mnt/output"); bad = 0
for cell in ['coqa_Llama2_13b', 'hotpotqa_Llama2_13b', 'nqopen_Llama2_13b', 'truthfulqa_Llama2_13b']:
    C = Cell(O, cell, "cal"); te = C.rows["test"]
    out = {"B2": C.U[te].mean(1), "B3-Sum": C.rawCP[te] + C.rawTK[te]}
    for m, u in out.items():
        a90 = official_auroc(u, C.acc[90][te], C.N); a70 = official_auroc(u, C.acc[70][te], C.N)
        r90, r70 = REF[(cell, m)]
        ok = abs(a90 - r90) < 5e-5 and abs(a70 - r70) < 5e-5; bad += (not ok)
        print(f"{cell:24s} {m:7s} tau90 {a90:.4f} (paper {r90:.4f})  tau70 {a70:.4f} (paper {r70:.4f})  {'OK' if ok else 'MISMATCH'}")
print("ALL OK" if bad == 0 else f"{bad} MISMATCH(ES)")

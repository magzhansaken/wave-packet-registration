#!/bin/bash
# Full reproduction on CPU. Set the slice folders first (see README):
#   D1_SLICES=data/d1 (main half), data/d1_reserve (reserve half), data/d1_dam (dam_D16); D3_SLICES=data/d3
set -e
gunzip -kf results/*.jsonl.gz 2>/dev/null || true   # per-unit rows of the original runs
export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
F=40,60,80,100,120,140,160,180,200,220,240,260
python3 -m pytest -q tests
python3 experiments/e1_numerics.py
python3 experiments/lemma_check.py
python3 experiments/e1_arrivals.py --solver
# main half: six frequencies, then the six confirmation frequencies
D1_SLICES=data/d1 python3 experiments/e2_run.py --workers 2 --freqs 40,60,100,140,180,220 --out results/e2_rows.jsonl
D1_SLICES=data/d1 E2_METHODS=bss_tr,bss_tr_gp python3 experiments/e2_run.py --workers 2 --freqs 40,60,100,140,180,220 --out results/e2_rows_bss_tr.jsonl
D1_SLICES=data/d1 E2_METHODS=br_gp python3 experiments/e2_run.py --workers 2 --freqs 40,60,100,140,180,220 --out results/e2_rows_brgp_fix.jsonl
D1_SLICES=data/d1 E2_NO_IN=1 python3 experiments/e2_run.py --workers 2 --freqs 80,120,160,200,240,260 --out results/e2_rows_confirm.jsonl
# reserve half and dam_D16
D1_SLICES=data/d1_reserve E2_NO_IN=1 python3 experiments/e2_run.py --workers 2 --freqs $F --out results/e2_rows_reserve.jsonl
D1_SLICES=data/d1_dam E2_NO_IN=1 python3 experiments/e2_run.py --workers 2 --freqs $F --out results/e2_rows_dam.jsonl
# ablations, mixing index, fatigue, neural baselines
D1_SLICES=data/d1 python3 experiments/e8_ablation.py
D1_SLICES=data/d1_reserve python3 experiments/e8c_confirm.py --rows results/e2_rows_reserve.jsonl --out results/e8c_rows_reserve.jsonl
D1_SLICES=data/d1_dam python3 experiments/e8c_confirm.py --rows results/e2_rows_dam.jsonl --out results/e8c_rows_dam.jsonl
D1_SLICES=data/d1 python3 experiments/e4b_mixing.py
D1_SLICES=data/d1 python3 experiments/e4_law.py
python3 experiments/e6_d3.py
D1_SLICES=data/d1 python3 experiments/e2_neural.py --workers 1
D1_SLICES=data/d1 NEURAL_SKIP_FVAE=1 NEURAL_SEEDS=0,1 python3 experiments/e2_neural.py --freqs 60,80 --out results/e2_neural_rows_band.jsonl
D1_SLICES=data/d1_dam OUT=results/dam_neural python3 experiments/e2_neural.py --workers 1 --out results/e2_neural_rows_dam.jsonl
D1_SLICES=data/d1_dam OUT=results/dam_neural NEURAL_SKIP_FVAE=1 NEURAL_SEEDS=0,1 python3 experiments/e2_neural.py --freqs 60,80 --out results/e2_neural_rows_dam_band.jsonl
# amendment 13: windowed shift-and-stretch, drift-aware conformal intervals, damage detection
D1_SLICES=data/d1 python3 experiments/e13_run.py --freqs 60,80,100 --rows results/e2_rows.jsonl,results/e2_rows_brgp_fix.jsonl,results/e2_rows_confirm.jsonl --out results/e13_rows_main.jsonl
D1_SLICES=data/d1_reserve python3 experiments/e13_run.py --freqs 60,80,100 --rows results/e2_rows_reserve.jsonl --out results/e13_rows_reserve.jsonl
D1_SLICES=data/d1_dam python3 experiments/e13_run.py --freqs 60,80,100 --rows results/e2_rows_dam.jsonl --out results/e13_rows_dam.jsonl
python3 experiments/e13_auc.py
python3 experiments/e13_diag.py
python3 experiments/timing_eval.py
# evaluation (numbers of all tables) and figures
for s in findings dam_evaluate bands_eval ablation_eval coverage_eval neural_eval d3_eval discussion_eval e8c_evaluate e13_eval; do
  python3 experiments/$s.py
done
for f in figures/fig_*.py; do python3 $f; done

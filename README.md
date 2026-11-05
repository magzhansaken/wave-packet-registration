# Bicharacteristic wave-packet registration for guided waves in composite plates

Code, configurations and per-unit results for the article *Bicharacteristic Wave-Packet Registration for Predicting
Guided Waves in Composite Plates beyond the Calibrated Temperature Range* (*Algorithms*, MDPI).

A pitch-catch record of a composite plate holds packets of several wave modes, and their travel times change with
temperature at different rates. The method predicts the record at a temperature outside the calibration range by
fitting a delay law in temperature to each packet window. The windows are placed at the packet arrival times computed
by a bicharacteristic-splitting solver of the laminate. Prediction intervals widen with the distance from the
calibration range. The repository also contains every baseline, the statistical evaluation and the scripts for all
tables and figures. Everything runs on CPU.

## Contents
* [Installation](#installation)
* [Quick start](#quick-start)
* [Data](#data)
* [Reproducing the article](#reproducing-the-article)
* [Tables and figures](#tables-and-figures)
* [Repository layout](#repository-layout)
* [Protocol and seeds](#protocol-and-seeds)
* [Citation](#citation) · [Licence](#licence)

## Installation
Python 3.13 (tested on Linux).
```bash
git clone https://github.com/magzhansaken/wave-packet-registration.git
cd wave-packet-registration
python3 -m venv .venv && source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```
Exact versions are pinned in `requirements.txt`: NumPy 2.5, SciPy 1.18, numba 0.68, scikit-learn 1.9, PyTorch 2.15
(CPU), bayesian-torch 0.5, h5py 3.16 and matplotlib 3.11. PyTorch and bayesian-torch are needed only for the neural
baselines.

## Quick start
These two steps need no downloaded data:
```bash
pytest -q tests                 # 19 tests, under a minute
python3 examples/quickstart.py  # a few seconds
```
The tests cover:
* the laminate and the characteristic decomposition;
* Lagrange interpolation and the order of the splitting;
* energy stability with traction-free edges and the Courant limit;
* symmetry and reciprocity;
* the windowed shift-and-stretch baseline, the weighted conformal quantile, ACI and AUC;
* the quick-start example.

The example builds synthetic records of S0- and A0-like packets that drift at different rates. It fits a single
tracked stretch and the wave-packet registration on 20–35 °C and predicts the record at 45 °C:
```
single stretch (tracked BSS)   relative L2 error at 45 C: 0.147
wave-packet registration       relative L2 error at 45 C: 0.054
```
The two predictors are called the same way as in the experiments:
```python
from surrogates import core
P, _ = core.fit_br_gp(t, S, T, T_query, "up", fk=60, packets=packets, kappa=kappa, lam=1.0, resid="gp")
```
Here `t` is the time axis, `S` holds the training records (n × m) at temperatures `T`, and `"up"` means that the
query temperatures lie above the training range. `packets` is a packet table from `bcs.rays.packet_table` and
`kappa` holds the pooled prior slopes per mode.

## Data
The data are open and are not redistributed here.
* **D1**: Open Guided Waves #1, a CFRP plate under temperature cycles, CC0 (doi:10.6084/m9.figshare.c.4488089;
  Moll et al., *Struct. Health Monit.* 2019, doi:10.1177/1475921718817169). The slicing script is
  `data_prep/slice_ogw1.py`: pitch-catch signals, every second measurement, 12 frequencies, float32 `.npz` per
  frequency and file. Set the environment variable `D1_SLICES` to the slice folder.
* **D3**: NASA CFRP composites fatigue data (Saxena et al., NASA Ames Prognostics Data Repository). The slicing script
  is `data_prep/slice_nasa_cfrp.py`; set `D3_SLICES` to its output folder.
* **Integrity checks** of the slices (counts, NaN, duplicates, temperature ranges, overlap between the sets):
  `data_prep/merge_check.py`, `check_reserve.py` and `check_dam.py`.

Set `RAW_PREFIX` to the folder with the downloaded raw files before slicing.

## Reproducing the article
1. Download OGW #1 and the NASA CFRP data, run the two slicing scripts and the integrity checks.
2. Run `pytest -q tests`.
3. Run `bash run_all.sh`. It holds the exact command lines of all experiments in the order they were run, then the
   evaluation scripts (`experiments/*_eval.py`) and the figures (`figures/fig_*.py`).

All per-unit rows of the original runs are in `results/` (`*.jsonl.gz`). After unpacking them, the evaluation
scripts recompute every number of the tables from these rows without the raw data:
```bash
gunzip -k results/*.jsonl.gz results/*/*.jsonl.gz 2>/dev/null
for s in findings dam_evaluate bands_eval ablation_eval coverage_eval neural_eval d3_eval discussion_eval e8c_evaluate e13_eval; do
  python3 experiments/$s.py; done
```
The full set of experiments takes many CPU hours; the neural baselines take most of them.

## Tables and figures
| Table / figure | Script | Result file |
|---|---|---|
| Figure 1 (overview), Figure 2 (registration on one path) | `figures/fig_pipeline.py`, `figures/fig_registration.py` | `fig1_registration.json` |
| Table 2, Figure 3 (data) | `data_prep/*.py`, `figures/fig_data.py` | `data_checks*.json`, `fig2_data.json` |
| Figure 4, Table 3 (solver against measurements) | `experiments/e1_arrivals.py --solver`, `e1_solver_compare.py`, `figures/fig_solver_signals.py` | `e1_arrivals.json`, `e1_solver_compare.json`, `e1_solver_T4_40k.npz` |
| Table 4, Figure 5 (gain by frequency band) | `experiments/bands_eval.py`, `figures/fig_bands.py` | `bands_eval.json` |
| Table 5 (windowed shift and stretch) | `experiments/e13_run.py`, `e13_eval.py` | `e13_eval.json` |
| Table 6 (ablation) | `experiments/e8_ablation.py`, `ablation_eval.py`, `e8c_confirm.py`, `e8c_evaluate.py` | `ablation_eval.json`, `e8c_eval.json` |
| Table 7, Figure 6, Table 8 (intervals) | `experiments/coverage_eval.py`, `e13_eval.py`, `figures/fig_coverage3.py` | `coverage_eval.json`, `e13_eval.json` |
| Section 4.8 (damage detection, AUC) | `experiments/e13_auc.py`, `e13_diag.py`, `e13_eval.py` | `e13_eval.json`, `e13_diag.json` |
| Table 9, Figure 7, Figure A2 (neural baselines, cost) | `experiments/e2_neural.py`, `neural_eval.py`, `timing_eval.py`, `figures/fig_cost.py` | `neural_eval.json`, `timing_eval.json` |
| Table 10 (fatigue) | `experiments/e6_d3.py`, `d3_eval.py` | `d3_eval.json` |
| Figure 8 (mode-mixing index) | `experiments/e4b_mixing.py`, `discussion_eval.py`, `figures/fig_mixing.py` | `discussion_eval.json` |
| Propositions 1–2, Lemma 1 | `experiments/theory_mc.py`, `theory_mc_impl.py`, `lemma_check.py` | `theory_mc*.json`, `lemma_check.json` |
| Table A1 (MF-B-NO on its authors' data) | `baselines/mfbno_authors/run_repro_val.py`, `experiments/mfbno_repro_eval.py` | `mfbno_repro_eval.json` |
| Figure A1, Table A2 (solver on exact solutions, cost of the steps) | `experiments/e1_numerics.py`, `figures/fig_solver_exact.py` | `e1_numerics.json` |
| Tables S1–S2 of the Supplementary Materials (configurations, protocol) | `configs/search_space.yaml`, `docs/PROTOCOL.md` | `config_choice.json` |

## Repository layout
```
bcs/            plate model and bicharacteristic-splitting solver
  laminate.py     ply stiffness, classical lamination theory (A, B, D, A_s)
  system.py       first-order plate system (membrane + Mindlin bending), energy symmetrizer, dispersion
  solver.py       splitting solver: characteristic sweeps, traction-free edges, Strang/Yoshida composition, PZT (numba)
  rays.py         group-velocity curves and packet arrival tables (direct and edge-reflected rays)
  ogw.py          OGW #1 loader, transducer coordinates, path geometry
surrogates/
  core.py         copy, kNN, PCA-GP, baseline signal stretch (plain and tracked), wave-packet registration (br, br_gp),
                  DTW and stretch oracles, split-conformal and growth-law intervals
  neural.py       MF-B-NO-type Bayesian DeepONet residual, f-VAE
  br2.py          registration with intercepts (post-hoc variant A2)
experiments/    one script per experiment (E1-E13) and per evaluation (*_eval.py)
figures/        one script per figure, shared style in style.py
data_prep/      slicing of the raw data and integrity checks
baselines/      reproduction of MF-B-NO on its authors' data
examples/       quick start on synthetic data
tests/          pytest suite
configs/        tuning grids of all methods
docs/           evaluation protocol and its amendments
results/        result files of all runs (per-unit rows as jsonl.gz, summaries as json)
```

## Protocol and seeds
The metric, the unit of analysis, the tests and the coverage threshold were fixed before the experiments. Every later
change is recorded in `docs/PROTOCOL.md` (amendments 1–13), with the SHA-256 prefix of the frozen code where a claim was
fixed before its run.

Seeds:
* MF-B-NO: 0, 1, 2 on the main half and 0, 1 on dam_D16;
* f-VAE: 0, 1, 2;
* bootstrap: generator seed set in `experiments/analyze.py`.

All other methods are deterministic. The tuning grids are in `configs/search_space.yaml` and in Table S1 of the Supplementary Materials of the
article.

## Citation
If you use this code, please cite the article (metadata also in `CITATION.cff`):
```bibtex
@article{Mukasheva2026wavepacket,
  author  = {Mukasheva, Togzhan and Zhuzbayev, Serik and Sarsenbay, Magzhan and Mukasheva, Asylzhan and Ilyasova, Meruert},
  title   = {Bicharacteristic Wave-Packet Registration for Predicting Guided Waves in Composite Plates beyond the
             Calibrated Temperature Range},
  journal = {Algorithms},
  year    = {2026}
}
```

## Licence
The code is under the MIT licence (`LICENSE`). The data keep the licences of the original data sets.

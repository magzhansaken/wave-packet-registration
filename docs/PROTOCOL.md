# Evaluation protocol and its amendments

The metric, the unit of analysis, the tests and the coverage threshold were fixed before the experiments
(2 November 2026). Every later change was written down, with its date and reason, before the run it affected.
Code hashes are SHA-256 (first 16 characters).

## Fixed before the experiments

| Item | Decision |
|---|---|
| Main metric | relative L2 error of the predicted signal outside the calibration range, on the analysis window 80 µs – 1 ms |
| Secondary metrics | coverage of nominal 95 % bands (share of signal samples inside the band); mean half-width relative to the RMS of the test signal |
| Unit of analysis | split × frequency × path; the metric is averaged over the test signals of the unit |
| Test | two-sided Wilcoxon signed-rank test over units; Holm correction within each family of claims; α = 0.05 |
| Effect | median of (e_base − e_ours)/e_base; 95 % bootstrap CI over units (10 000 resamples) and over whole splits (2 000) |
| Claim criterion | p_Holm < 0.05 and the unit CI entirely above zero |
| Coverage threshold | ≥ 0.90 in every extrapolation block (split; averaged over frequencies and paths) |
| Seeds | neural models: spread of the model only, never a p-value |
| Tuning | every method: at most 12 configurations, chosen by the median validation error on 12 fixed paths; validation block = outer quarter of the training block; inner part ends 2.5 °C before it |
| Splits (temperature) | train ≤ 30/35/40 °C → test ≥ 35/40/45 °C; train ≥ 40/45/50 °C → test ≤ 35/40/45 °C (5 °C gap) |
| Splits (fatigue) | train on the first half of the cycle levels of a coupon, skip one level, test on the rest |

## Amendments

| No. | Date | What was fixed before the run | Outcome |
|---|---|---|---|
| 1–3 | 2 Nov | use of every second of the 322 intact measurements (main half), the other half kept unopened (reserve half); the wavefield data set dropped (file size) | — |
| 4 | 3 Nov | validation gap 2.5 °C; tuning per split and frequency on 12 fixed paths; zero-mean GP residual for all residual models; window labels by the last direct arrival | — |
| 5 | 3 Nov | neural baselines on a reduced set of units | — |
| 6 | 3 Nov | tracked stretch added as a stronger stretch baseline; the strongest stretch baseline = the better of the two on validation | — |
| 7 | 3 Nov | confirmation pass on the six frequencies not used in development (80, 120, 160, 200, 240, 260 kHz); ablations on 12 paths | — |
| 8 | 3 Nov | leak in the validation step found (pooled slopes and stiffness factor from the whole training block); all fits evaluated on validation now use the inner part only; registration rerun | rerun replaces the earlier rows |
| 9 | 3 Nov | neural baselines: 36 units (2 fixed paths × 6 splits × 40, 100, 180 kHz) | — |
| 10 | 3 Nov | solver-only rule for "registration or stretch", criteria S1/S2 on the reserve half | not met (ρ = 0.13 against 0.42 for the index from training data) |
| 11 | 4 Nov | clean test on dam_D16 before the data were opened: band claim (60–100 kHz) and criteria; frozen code `surrogates/core.py` cd9d0f0887ce7e36, `experiments/e2_run.py` 5c123927f0997477 | band claim confirmed |
| 12 | 4 Nov | hypothesis H15: ray windows beat uniform windows in the band, on the reserve half and dam_D16; `experiments/e8c_confirm.py` 61b8ae44452792ea | confirmed on both sets |
| 13 | 4 Nov | (13.1) AUC of damage detection on dam_D16 after compensation; (13.2) windowed shift-and-stretch baseline; (13.3) nexCP and ACI intervals; (13.4) MF-B-NO at 60 and 80 kHz; frozen `experiments/e13_run.py` 3c4376703e281cfc, `experiments/e13_auc.py` (after the fix of the configuration file list) | windowed stretch, conformal methods and MF-B-NO: claims met in the band on all sets; AUC: not better than the stretch (both at chance). Comparisons reported on the 216 band units per set; the runs outside the band were stopped |

## Assumptions of the plate model
Transverse shear moduli G12 = G13 = 4.2 GPa and G23 = 3.0 GPa (nominal values of the prepreg class); one stiffness
factor fitted to S0 arrivals at 60 and 100 kHz on the training block; temperature law of the matrix-dominated constants
fitted to A0 delays at 40 kHz between 20 and 40 °C.

## Post-hoc analyses (not part of any recorded test)
* **A2** — registration with an intercept in every delay law (`surrogates/br2.py`, `experiments/e2_run_a2.py`,
  `experiments/a2_compare.py`, `experiments/a2_diag.py`). Motivation: at the same plate temperature, records from the
  heating and the cooling legs of the cycle differ by a small delay offset, which a law through the edge record
  cannot absorb. Reported in the ablation (Table 6 of the article) as `br_gp2` and marked post hoc there.
* **Pilot analysis** — the 40 kHz, pair-1 subset of OGW #1 used before the protocol was fixed; it enters only the
  overlap check of `data_prep/merge_check.py`.

## Frozen files in this release
The SHA-256 prefixes above are those of the files at the time each amendment was fixed. In the released copy, the
module docstrings of `experiments/e8c_confirm.py` and `experiments/e13_run.py` point to this file instead of the
internal protocol file name; no code line differs. The prefixes of the released files are:

| File | Frozen | Released |
|---|---|---|
| `surrogates/core.py` | cd9d0f0887ce7e36 | cd9d0f0887ce7e36 |
| `experiments/e2_run.py` | 5c123927f0997477 | 5c123927f0997477 |
| `experiments/e8c_confirm.py` | 61b8ae44452792ea | 7c3ae89461de4a26 |
| `experiments/e13_run.py` | 3c4376703e281cfc | 144d673796361e64 |

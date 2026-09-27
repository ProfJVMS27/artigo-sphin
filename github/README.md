# SPHIN: Sobol-Physics-Informed NSGA-II

Reproducibility package for the SPHIN radiation-shielding case study. The workflow generates feasible seven-layer configurations, evaluates the high-fidelity MCNP6 model, trains surrogate models, compares adaptive and canonical NSGA-II, and validates pre-specified Pareto solutions with independent MCNP6 calculations.

## Repository layout

- `src/`: phase 1--4 scripts as used in the study.
- `data/`: 2,000 Sobol+FAM configurations and the processed DE/DF AP MCNP6 dataset.
- `models/`: frozen PLS model artifact used for the timing benchmark.
- `results/`: surrogate and MCNP6 validation result tables.
- `manuscript/`: LaTeX source and bibliography.
- `docs/`: data provenance and release instructions.

## Reproducibility notes

1. MCNP6 is proprietary/licensed software and is not distributed here. Configure the path to a licensed installation before running `src/phase2_mcnp_campaign.py`.
2. The stored raw `outp` and `runtpe` files are intentionally excluded from Git because of their volume. Archive them in Zenodo/OSF and record the release DOI in `docs/DATA_PROVENANCE.md` before public release.
3. The campaign uses the radiological-only geometry, no grenade component, a 121 cm source--detector center distance, W--epoxy density 4.643 g/cm$^3$, and F4:P coupled with DE4/DF4 ICRP 116 AP coefficients.
4. The source scripts preserve the original operational file layout. Update local paths in the configuration block before running them from another machine.

## Environment

Create a Python environment and install the packages in `requirements.txt`. The published scripts were run with a Windows MCNP6 installation and Python scientific/ML packages. The package versions should be pinned to the authors' final environment export before release.

## Timing benchmark

Run `python src/benchmark_pls_inference.py`. It reports model-only inference times for the selected Elastic Net and the PLS benchmark; it does not include feature construction or analytical objective calculations. The manuscript comparison uses the 14 h, 2,000-case campaign with three concurrent MCNP6 jobs and the median of five 1,000-call batches.

## Citation and release

Add repository URL, Zenodo DOI, license, author list, and `CITATION.cff` metadata before making the repository public. Do not upload institutional paths, MCNP6 binaries, or files that are subject to a third-party license.

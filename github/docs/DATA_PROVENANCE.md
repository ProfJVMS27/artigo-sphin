# Data provenance and release checklist

## Included files

| File | Contents |
|---|---|
| `data/sobol_configurations_2000.csv` | Feasible seven-layer configurations generated in phase 1. |
| `data/mcnp_dedf_ap_processed_2000.csv` | Processed results from the corrected DE/DF AP MCNP6 campaign. |
| `results/mcnp_validation_24_original.csv` | Original validation extraction retained for audit traceability. |
| `models/PLS_Regression.joblib` | Frozen PLS artifact used for the model-only timing benchmark. |

## Before release

- [ ] Add the final 21-case validation table, retaining the original 24-case table as an audit artifact.
- [ ] Upload the raw MCNP input/output archive to Zenodo or OSF, then add its immutable DOI here.
- [ ] Replace local institutional paths in scripts with repository-relative configuration.
- [ ] Add author names, an approved license, and `CITATION.cff`.
- [ ] Confirm that material-price reports may be redistributed; otherwise retain citations and do not upload source PDFs.
- [ ] Create a tagged GitHub release and connect it to Zenodo to mint a versioned DOI.

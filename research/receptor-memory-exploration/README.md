# Exploration scripts (receptor-field memory)

Raw scripts used while developing `mycelia/receptor_memory.py`. Kept for
traceability; use `tools/receptor_memory_mri.py` to reproduce results.

- `load.py`: cache images at a resolution with the clean split.
- `exp.py`, `exp2.py`: dictionary size / resolution / pooling sweeps (1-NN, centroid).
- `ridge.py`, `rbf2.py`, `ens.py`: readout comparisons (these printed Testing accuracy during exploration).
- `exp3.py`, `exp4.py`: mirror and shifted imprint encodings.
- `final.py` -> `final.json`: validation-selected settings, Testing scored once (94.48%).

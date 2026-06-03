Script organization:

- `train/`: scan submission and worker entrypoints.
- `evaluate/`: per-run or per-scan diagnostics written into result folders.
- `visualize/`: scan-wide aggregate figures.
- `collect/`: tabular scan summaries.
- `benchmark/`: standalone benchmark utilities.

Common commands:

```bash
python3 scripts/evaluate/evaluate_energies.py
python3 scripts/evaluate/evaluate_observables.py
python3 scripts/evaluate/evaluate_one_body_density_matrix.py --run-dir <result-folder>
python3 scripts/visualize/plot_scan_diagnostics.py
python3 scripts/visualize/plot_scan_observables.py
python3 scripts/collect/summarize_scan_results.py
python3 scripts/benchmark/benchmark_ewald.py
```

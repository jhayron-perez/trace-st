# TraCE-ST processed data and trajectory ensembles

## Overview

This deposit contains the analysis-ready gridded inputs, trajectory ensembles, hyperparameter summaries, and intermediate result tables supporting:

> Pérez-Carrasquilla, J. S., Nichol, J. J., Robledo, V., Bull, D., Dagon, K., Evans, M. N., & Molina, M. J. (2026). Tracing the space-time causal origins of Earth system extremes.

TraCE-ST (Tracer of Causal Evolutions in Space and Time) reconstructs event-conditioned causal pathways in multivariate gridded data by iteratively tracing lagged causal parents through localized M-CaStLe graphs. These files support the controlled synthetic experiments and the three real-world applications presented in the manuscript.

Code is maintained at [jhayron-perez/trace-st](https://github.com/jhayron-perez/trace-st).

## Directory structure

```text
real_world_inputs/
results_debby/
results_pinatubo/
results_pnw21/
results_synth1/
results_synth2/
README.md
manifest.json
```

### `real_world_inputs/`

Analysis-ready, compressed `float32` NetCDF inputs for the real-world cases. Each file contains one data variable, `trace_st_data`, with dimensions ordered as:

```text
time × var × lat × lon
```

Longitudes use degrees east on `[0, 360)`. Files are temporally and spatially restricted to the domains needed for the manuscript analyses.

- `trace_st_debby_processed.nc`: brightness temperature (`Tb`), 500–700-hPa relative vorticity (`vo`), and precipitation (`precip`), hourly.
- `trace_st_pinatubo_processed.nc`: SO2 burden (`TMSO201`), sulfate aerosol burden (`BURDENSO401`), H2SO4 burden (`TMH2SO401`), and visible aerosol optical depth (`AEROD_v`), daily.
- `trace_st_pnw21_processed.nc`: 500-hPa geopotential (`Z500`), 10-hPa geopotential (`Z10`), outgoing longwave radiation (`MTNLWRF`), total column water vapor (`TCWV`), mean surface latent heat flux (`MSLHF`), and mean surface sensible heat flux (`MSSHF`), daily.
- `manifest.json`, when present in this directory, records dimensions, variables, byte sizes, and SHA-256 hashes generated during export.

Independent reference data such as the Debby IBTrACS track and Mount Pinatubo coordinates are used for physical validation and figure annotation, not as causal-discovery inputs.

### `results_debby/`

Hyperparameter summaries, member-level diagnostics, run metadata, and compressed trajectory bundles for Tropical Storm Debby (2006). Bundle pairs contain trajectory arrays (`.npz`) and their parameters/metadata (`.json`). These files support the Debby trajectory, storm-track-distance, causal-relevance, spatial-density, and supplementary ensemble analyses.

### `results_pinatubo/`

Hyperparameter summaries, member-level diagnostics, run metadata, and compressed trajectory bundles for the Mount Pinatubo eruption (1991). These files support the endpoint-distance-to-volcano, variable-relevance, spatial-density, and supplementary ensemble analyses.

### `results_pnw21/`

The PNW heatwave hyperparameter-search summary and retained run directories. Each retained run contains its parameters, summary, and ensemble trajectories. These files support pooled trajectory density, lead-time-resolved variable relevance, physical-pattern composites, and supplementary ensemble examples.

Python pickle files are included because the manuscript analysis code reads them directly. They should be opened only in a trusted environment and with package versions compatible with the repository environment.

### `results_synth1/`

Trial-level and trajectory-level results for the controlled two-variable experiment. A prescribed causal feature propagates in `V1`, while `V2` is correlated with `V1` but does not cause it. The files support causal-path reconstruction, parent-selection, robustness, and causal-discovery-backend comparisons.

### `results_synth2/`

Trial-, mixture-, and member-level results plus compressed trajectory bundles for the controlled three-variable experiment. Prescribed pathways in `V1` and `V3` jointly produce target `V2` according to `alpha_mix`. The files support recovery of competing pathways and their relative causal contributions. Results may be separated by Elastic-Net Granger, PCMCI, and DYNOTEARS backends.

## Common result files

Depending on the experiment, result directories contain:

- `results_jobs_final.csv`: one row per completed hyperparameter configuration;
- `results_tracks_final.csv`: trajectory-level results for the two-variable synthetic experiment;
- `results_alphas_final.csv`: summaries by prescribed `alpha_mix` value;
- `results_members_final.csv`: member-level completion, geometry, parent, and attribution diagnostics;
- `run_metadata.json`: execution configuration and search metadata;
- `run_summary.json`: aggregate completion and runtime information;
- `bundles/*.npz`: padded trajectory centers, parent sequences, lengths, completion flags, densities, and case-specific arrays;
- `bundles/*.json`: parameters and metadata corresponding to each `.npz` bundle;
- `hyperparam_search_summary.csv`: PNW trial-level summary;
- `run_*/params.json`, `run_*/summary.json`, and `run_*/ensemble_results.pkl`: PNW retained-run artifacts.

Restart files such as `results_*.jsonl` and `results_*_snapshot.csv` are operational checkpoints and are not required when complete final tables are present.

## Minimal loading examples

Load an analysis-ready input:

```python
import xarray as xr

ds = xr.open_dataset("real_world_inputs/trace_st_debby_processed.nc")
data = ds["trace_st_data"]
print(data.dims, data.coords["var"].values)
```

Load a portable trajectory bundle:

```python
import json
import numpy as np

arrays = np.load("results_debby/bundles/debby_trial_00000.npz")
with open("results_debby/bundles/debby_trial_00000.json") as stream:
    metadata = json.load(stream)
```

Exact bundle names and retained trial identifiers are listed in the corresponding final CSV tables.

## Reproducibility

The scripts and figure-analysis notebooks are in the repository’s `Scripts_Paper/` directory. Create the software environment and install the package with:

```bash
conda env create -f trace-st.yml
conda activate trace-st
pip install -e .
```

File paths in the research scripts reflect the original high-performance-computing environment and should be replaced with the local paths to this deposit. The final CSV files should be preferred over snapshot or JSONL restart files.

## Data provenance

- Debby precipitation: GPM IMERG Final product, Version 07.
- Debby brightness temperature: NCEP/CPC merged infrared brightness-temperature product.
- Atmospheric fields: ERA5 reanalysis.
- Debby observed track: IBTrACS North Atlantic archive.
- Pinatubo fields: E3SMv2-SPA simulation described in the manuscript.
- PNW heatwave fields: ERA5-derived standardized daily anomalies described in the manuscript.

Users are responsible for following the attribution and licensing requirements of the original data providers. The processed files in this deposit are scientific derivatives prepared for reproducing the TraCE-ST analyses.

## Contact

Jhayron S. Pérez-Carrasquilla  
Department of Atmospheric and Oceanic Science, University of Maryland  
jhayron@umd.edu; jspecar@gmail.com

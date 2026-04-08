# TraCE-ST: Tracing the Space–Time Causal Origins of Earth System Extremes

This repository contains the code used in:

**“Tracing the space–time causal origins of Earth system extremes”**  
J. S. Pérez-Carrasquilla et al. (Science Advances, under review)

---

## Overview

Understanding the causes of Earth system extremes is challenging due to:
- the absence of controlled experiments,
- high-dimensional and nonlinear dynamics,
- and limitations of traditional causal discovery methods.

This repository implements **TraCE-ST (Tracer of Causal Evolutions in Space and Time)**, a data-driven framework designed to reconstruct event-conditioned causal pathways in multivariate spatiotemporal data.

TraCE-ST moves beyond static causal graphs by producing **causal trajectories**, representing the sequence of spatial and cross-variable influences leading to an event.

---

## Method Summary

TraCE-ST combines:

1. Local causal discovery (M-CaStLe)  
2. Directional clustering (DBSCAN)  
3. Lagrangian backtracking  
4. Probabilistic ensemble framework  

The output is an ensemble of causal trajectories, whose spatial density reflects the relative causal relevance of different regions and processes.

---

## Repository Structure

trace-st/
├── trace_st/                  # Core implementation of TraCE-ST
├── Scripts_SciAdv_Paper/      # Scripts used for experiments and figures in the paper

---

## Applications

TraCE-ST is validated across:

### Synthetic experiments
- Recovery of prescribed causal trajectories
- Robustness across causal discovery methods and hyperparameters

### Real-world case studies
- Tropical Storm Debby (2006)
- Mount Pinatubo eruption (1991)
- Pacific Northwest heatwave (2021)

---

## Installation

git clone https://github.com/jhayron-perez/trace-st.git
cd trace-st

pip install -r requirements.txt

---

## Usage

Typical workflow:

1. Prepare multivariate gridded data (e.g., xarray)
2. Define:
   - event location and time
   - target variable
   - spatial region and time window
3. Run TraCE-ST to generate trajectories

Example scripts are available in:
Scripts_SciAdv_Paper/

---

## Notes

- Results depend on variable selection, resolution, and hyperparameters
- TraCE-ST is intended as a hypothesis-generation tool

---

## Citation

Pérez-Carrasquilla, J. S., et al.  
Tracing the space–time causal origins of Earth system extremes.  
To be submitted.

---

## License

MIT License

---

## Contact

Jhayron S. Pérez-Carrasquilla  
Department of Atmospheric and Oceanic Science  
University of Maryland

# Spatiotemporal Nematode Population Modelling

This repository houses a numerical simulation framework written in Python to track and predict the spatial and temporal spread of potato cyst nematodes (PCN) across agricultural fields. The system integrates deterministic logistic population growth with stochastic Poisson processes and continuous Gaussian spatial kernels to evaluate and optimise targeted soil sampling strategies over a 10-year timeline. This project was developed as a university research project and achieved a first-class grade of 70%.

## Structural Overview

Traditional grid-based soil sampling yields low initial detection rates because nematode hotspots cluster tightly rather than scattering uniformly. To resolve this testing inefficiency, this framework maps population dynamics using two distinct operational modes:

* **Mode 1 (Expansion Prediction):** Simulates continuous biological growth and outwards vector dispersal from existing, confirmed hotspot coordinates to forecast active invasion frontiers.
* **Mode 2 (Risk-Based Assessment):** Estimates an introductory probability surface for untested fields using localized vectors (field entrance coordinates, neighbouring infestations, crop rotation frequency, and soil mechanics) to prioritize initial survey points.

## Core Repository Structure

* **main.py:** The primary execution dashboard containing a console menu interface to run each simulation component.
* **baseline_gmm.py:** Handles the multi-component Gaussian Mixture Model grid setup to represent initial clustered hotspot distributions.
* **risk_surface_assessment.py:** Processes the spatial vector weights, calculates individual factor probabilities, and classifies fields into four priority testing zones.
* **pipeline_handoff.py:** Demonstrates the integrated system architecture, transitioning from a static risk map to active temporal expansion predictions following simulated physical discovery.
* **sampling_efficiency.py:** Computes detection yield and cost-effectiveness curves (£ per hotspot detected) to compare traditional grid configurations against targeted frontier strategies.
* **sensitivity_analysis.py:** Tests model stability by isolating how changing expansion rates (v), seasonal growth rates (r), and hotspot frequencies (lambda) alter long-term area infestation metrics.
* **visualisations.py:** Handles the core greyscale contour plotting engine to ensure figures remain fully legible when printed in black and white.

## Mathematical Model Foundation

The underlying engine relies on three mathematical components calculated natively via vectorised arrays:

1. **Logistic Population Dynamics:** Controls localized population density scaling using a standard non-linear differential equation capped at the environmental carrying capacity (K).
2. **2D Gaussian Dispersal Spatial Kernels:** Handles outward continuity, mapping smooth density decay as radial distance from the introduction centre increases.
3. **Independent Poisson Distribution:** Handles the discrete probability mass function for structural hotspot generation across a multi-field system.

## Requirements

The platform requires Python 3 and the standard tracking packages listed below:

* **numpy** (matrix slicing and numerical arrays)
* **scipy** (integration and mathematical solvers)
* **scikit-learn** (unsupervised mixture modelling components)
* **matplotlib** (spatial coordinate plotting and contours)

## Execution

To launch the local platform interactive console menu, execute the main entry file via your terminal window:

```bash
python main.py

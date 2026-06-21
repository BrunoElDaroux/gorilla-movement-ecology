# 🦍 Spatial Movement Ecology & Intergroup Dynamics in Mountain Gorillas

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/BrunoElDaroux/gorilla-zoonotic-analysis/main)

> **Study Area:** Virunga Massif, Rwanda / DRC / Uganda
> **Species:** *Gorilla beringei beringei* (Mountain Gorilla)
> **Tools:** Python · GeoPandas · SciPy · Matplotlib · Seaborn \n
> **Data:**  SMART Conservation Software GPS Export

---

## 📋 Project Overview

This project analyzes GPS movement data from habituated mountain gorilla groups in the Virunga Massif to characterize:
- Daily movement distances and temporal consistency
- Home range estimates (MCP and KDE methods)
- Inter-group spatial separation and avoidance behavior
- Nest-site predictability for field monitoring efficiency

GPS fixes are recorded every 10 minutes by field researchers walking with gorilla groups from morning nest departure (~06:00) to mid-afternoon (~14:00), simulating SMART Conservation Software exports.

---

## 🦍 Research Questions & Key Findings

---

### 🔬 Research Questions

The project was built around four core questions:

**1. How do mountain gorilla groups move through the Virunga Massif?**
What are the daily path lengths, movement speeds, and directional patterns — and are they consistent enough to be useful for field operations?

**2. How large are the home ranges of each group, and do they vary by season?**
Using both MCP and KDE methods — what territory does each group occupy, and does rainfall season affect ranging behavior?

**3. Do gorilla groups actively maintain spatial separation from one another?**
This was the central hypothesis test: are observed inter-group distances greater than what would occur if groups moved independently with no awareness of each other?

**4. How predictable is a group's daily location from its previous night's nest site?**
A question with direct conservation and operational value — can field rangers reliably relocate groups each morning?

---

### 📊 Key Findings Summary

**Finding 1 — Daily movement is short and temporally consistent**
The mean daily path length was **1.30 ± 0.19 km** across all six groups, with a coefficient of variation of just 0.15. This low variability means group movement is highly routine and predictable from day to day. Pablo was the most active group (1.51 km/day), Umubano the least (1.12 km/day), consistent with group size differences.

**Finding 2 — Nest sites are highly predictable**
The mean nest-to-nest distance between consecutive days was only **0.40 km**. Critically, **70% of all nightly transitions placed the group within 500 meters of the previous nest site**, and 99% were within 1 km. This means field researchers can begin each morning's follow from the previous evening's nest location with very high confidence of finding the group — a direct operational gain for monitoring efficiency and ranger logistics.

**Finding 3 — Groups actively maintain spatial separation (H₀ rejected)**
This was the statistically strongest finding. Using a permutation null model where each group independently drew random positions from its own empirical location cloud (simulating movement with no intergroup awareness), the null expected minimum pairwise distance was **1.54 km**. The observed mean minimum distance was **4.54 km** — a difference of **+3.00 km**. The permutation test returned **p < 0.001** with a **Cohen's d of 2.05**, a large effect by any standard. All three tests (permutation, t-test, Mann-Whitney U) agreed. H₀ was rejected.

**Finding 4 — Spatial avoidance reflects behavioral regulation of group stability**
The biological interpretation of Finding 3 is that gorillas are not simply wandering independently — they are actively tracking and avoiding neighboring groups. Intergroup encounters in mountain gorillas carry high costs: silverback dominance contests risk serious injury, females may switch groups (a reproductive cost to resident males), and the energetic disruption cascades across the group. Spatial avoidance is therefore a behavioral mechanism that stabilizes group composition and reduces conflict.

**Finding 5 — Home ranges are moderate and show seasonal variation**
KDE 95% utilization distributions ranged from roughly 18–34 km² depending on group. Larger groups (Susa, Pablo) used larger ranges, consistent with published literature on group-size range scaling. Home ranges contracted slightly during the long rains season, likely reflecting higher food availability reducing the need to travel.

---

### 🔑 One-Line Takeaway for Each Bullet 

| CV Bullet | Core Finding |
|---|---|
| GPS movement characterization | Mean 1.30 km/day, low CV — short and consistent |
| Home range + intergroup metrics | 18–34 km² KDE ranges; 4.54 km mean group separation |
| Hypothesis testing | p < 0.001, d = 2.05 — active avoidance confirmed |
| Nest-site predictability | 70% within 500m — groups reliably relocatable |
| Visualizations for reporting | 16 publication-ready figures across all analyses |

> ⚠️ **Data Note:** The dataset used in this project `(gorilla_gps_data.csv)` is synthetically generated to replicate the structure and statistical properties of real mountain gorilla GPS tracking data recorded via SMART Conservation Software. Real data was not published here due to conservation sensitivity — precise location data for critically endangered species is typically restricted to prevent poaching risk. The synthetic data preserves biologically realistic movement parameters (daily path lengths, home range sizes, intergroup distances) drawn from peer-reviewed literature. Researchers with institutional affiliation may request access to real Virunga gorilla tracking data through Movebank or directly from the Dian Fossey Gorilla Fund.
---

## 📁 Project Structure

```
gorilla-movement-ecology/
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   ├── raw/gorilla_gps_data.csv       <- GPS tracking data (generated by src/data_generator.py)
│   ├── processed/                     <- Cleaned datasets (auto-generated by notebooks)
│   └── README_data.txt
├── src/
│   ├── __init__.py
│   ├── data_generator.py              <- Synthetic data generator
│   ├── movement_metrics.py            <- Distance, speed, turning angle calculations
│   ├── home_range.py                  <- MCP and KDE home range estimation
│   ├── intergroup_analysis.py         <- Pairwise distance and overlap metrics
│   └── visualization.py              <- Plotting utilities
├── notebooks/
│   ├── 01_data_loading_exploration.ipynb
│   ├── 02_movement_analysis.ipynb
│   ├── 03_home_range_estimation.ipynb
│   ├── 04_intergroup_dynamics.ipynb
│   ├── 05_hypothesis_testing.ipynb
│   └── 06_final_visualizations.ipynb
└── outputs/
    ├── figures/                       <- All saved plots
    └── reports/                       <- Summary CSVs and tables
```

---

## 📊 Dataset

### Primary Source (Real Data)
- **Movebank**: https://www.movebank.org  
  Search: "Mountain gorilla Gorilla beringei" — free registration required
- **GBIF**: https://www.gbif.org/species/2889338
- **Dryad**: Grueter et al. (2022) gorilla ranging data

### Simulated Groups (Real Virunga Groups Used as Templates)
| Group   | Size | Center Lat | Center Lon |
|---------|------|------------|------------|
| Susa    | 28   | -1.435     | 29.522     |
| Hirwa   | 13   | -1.461     | 29.538     |
| Amahoro | 17   | -1.478     | 29.558     |
| Umubano | 12   | -1.452     | 29.571     |
| Pablo   | 24   | -1.490     | 29.542     |
| Kwitonda| 18   | -1.505     | 29.525     |

---

## 🚀 Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset
python src/data_generator.py

# 3. Open VS Code and run notebooks in order (01 -> 06)
code .
```

Install VS Code Jupyter extension: ms-toolsai.jupyter

---

## 📚 References

1. Robbins, M.M. et al. (2009). Intergroup encounters among mountain gorillas. *Am. J. Primatol.*
2. Grueter, C.C. et al. (2013). Nest site selection in mountain gorillas. *Behav. Ecol. Sociobiol.*
3. Seaman & Powell (1996). Kernel home range estimation. *Ecology.*
4. IUCN SSC Primate Specialist Group (2023). Mountain Gorilla Status Survey.

## Author

*Bioinformatician with research experience at the Dian Fossey Gorilla Fund, building end-to-end computational pipelines across four domains: spatial movement ecology (GeoPandas, KDE, permutation testing), population genetics (CERVUS microsatellite LOD scoring, Queller-Goodnight kinship estimation), machine learning survival analysis (Random Forest, temporal cross-validation), and conservation epidemiology (logistic regression, SciPy hypothesis testing, temporal linkage). Technical stack: Python · R · SQL · scikit-learn · SciPy · GeoPandas · Git. All work is grounded in longitudinal biological datasets with direct conservation policy implications across the Virunga Massif — Rwanda, Uganda, and DRC.*

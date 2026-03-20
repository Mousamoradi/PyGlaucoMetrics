# PyGlaucoMetrics

**PyGlaucoMetrics** is an open-source, pure-Python package for glaucoma detection using visual field (VF) data — no R or rpy2 dependency required. It accepts Humphrey Field Analyzer (HFA) 24-2 and 10-2 test patterns and provides a full pipeline from raw VF data to ensemble glaucoma classification with an interactive GUI.

[![PyPI version](https://badge.fury.io/py/PyGlaucoMetrics.svg)](https://pypi.org/project/PyGlaucoMetrics/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Features

- **Pure Python** — all R/rpy2 calls replaced with `numpy`, `scipy`, and `pandas`
- **VF grid plots** — Sensitivity (grayscale), Total Deviation (TD), and Pattern Deviation (PD) with probability-based colourmaps matching the R `visualFields` convention
- **Five glaucoma classifiers**: UKGTS, LoGTS, Foster, Kangs, HAP2 (part 1 & 2)
- **Weighted ensemble decision** with GL/Non-GL probability scoring
- **Interactive PyQt5 GUI** — load a CSV dataset, enter a patient index, and instantly view Sensitivity / TD / PD plots alongside classifier predictions
- **Save results** — exports all plots (PNG with embedded colourmaps), classifier CSVs, and summary bar charts, all prefixed with the original dataset filename

---

## Installation

**Windows:**
```bash
pip install PyGlaucoMetrics[windows]
```

**Linux / macOS:**
```bash
pip install PyGlaucoMetrics
```

---

## Quick Start

```python
import pandas as pd
from PyGlaucoMetrics import visualFields

# 1. Load raw VF data (columns: id, eye, date, age, s1…s54)
df_vf = pd.read_csv('VF_Data.csv')

# 2. Compute TD, TDP, PD, PDP and global indices
df_td, df_tdp, df_gi, df_gip, df_pd, df_pdp, gh = visualFields.getallvalues(df_vf)

# 3. Plot a single exam (Sensitivity / TD probability / PD probability)
visualFields.vfplot(df_vf.iloc[[0]], type='s')    # sensitivity
visualFields.vfplot(df_vf.iloc[[0]], type='tds')  # TD probability
visualFields.vfplot(df_vf.iloc[[0]], type='pds')  # PD probability
```

### Launch the GUI

```bash
python GL_prediction.py
```

Or from Python:

```python
from PyGlaucoMetrics.GL_prediction import MainWindow
from PyQt5.QtWidgets import QApplication
import sys

app = QApplication(sys.argv)
w = MainWindow()
w.show()
sys.exit(app.exec_())
```

---

## Input Format

Your CSV must contain at minimum:

| Column | Description |
|--------|-------------|
| `id` | Patient/exam identifier |
| `eye` | `OD` or `OS` (also accepts `right`/`left`, `1`/`0`) |
| `date` | Exam date (string or datetime) |
| `age` | Patient age at exam |
| `s1`–`s54` | Sensitivity values in dB (HFA 24-2, 54 points) |

Column names are case-insensitive (`ID`, `Age`, `Eye` are all accepted).

---

## Tested Datasets

| Dataset | Description |
|---------|-------------|
| `vfpwgRetest24d2` | Short-term retest data from 30 glaucoma patients (Queen Elizabeth Health Sciences Centre, Halifax, NS); 12 weekly sessions |
| `vfpwgSunyiu24d2` | 24-2 data from a single glaucoma patient (age anonymised) |
| `vfctrSunyiu24d2` | Healthy-eye normative data for 24-2 (courtesy W.H. Swanson & M.W. Dul) |
| `vfctrSunyiu10d2` | Healthy-eye normative data for 10-2 (courtesy W.H. Swanson) |

---

## Classifiers

| Classifier | Input | Criterion |
|------------|-------|-----------|
| **UKGTS** | TD probabilities | ≥2 consecutive points with p ≤ 0.01 |
| **LoGTS** | TD values | ≥2 points with TD < −10 dB |
| **Foster** | PD probabilities | Hemifield asymmetry + ≥3 points p ≤ 0.05 |
| **Kangs** | TD values | ≥3 points with TD < −5 dB |
| **HAP2** | PD probabilities + MD | Part 1: GL flag; Part 2: Stage 1/2/3 severity |

Final decision is a **weighted ensemble** combining all five classifiers with inverse-frequency class weights.

---

## Requirements

**Core (all platforms):**
```
numpy
pandas
matplotlib
scipy
PyQt5>=5.15
Pillow
seaborn
pingouin
requests
```

**Windows only:**
```
pywin32
```
Installed automatically with `pip install PyGlaucoMetrics[windows]`.

---

## Citation

If you use PyGlaucoMetrics in your research, please cite:

1. Moradi, M., Hashemabad, S.K., Vu, D.M., Soneru, A.R., Fujita, A., Wang, M., Elze, T., Eslami, M. and Zebardast, N. (2025). PyGlaucoMetrics: a stacked weight-based machine learning approach for glaucoma detection using visual field data. *Medicina*, 61(3), 541. https://www.mdpi.com/1648-9144/61/3/541

2. Moradi, M., Eslami, M., Hashemabad, S.K., Friedman, D.S., Boland, M.V., Wang, M., Elze, T. and Zebardast, N. (2024). PyGlaucoMetrics: An Open-Source Multi-Criteria Glaucoma Defect Evaluation. *Investigative Ophthalmology & Visual Science*, 65(7), OD38. https://iovs.arvojournals.org/article.aspx?articleid=2800368

3. Eslami, M., Kazeminasab, S., Sharma, V., Li, Y., Fazli, M., Wang, M., Zebardast, N. and Elze, T. (2023). PyVisualFields: A Python Package for Visual Field Analysis. *Translational Vision Science & Technology*, 12(2), 6. https://tvst.arvojournals.org/article.aspx?articleid=2785341

---

## License

MIT License — see [LICENSE](LICENSE) for details.

## Links

- **PyPI**: https://pypi.org/project/PyGlaucoMetrics/
- **GitHub**: https://github.com/Mousamoradi/PyGlaucoMetrics
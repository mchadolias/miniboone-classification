# MiniBooNE Particle Classification

![Tests](https://github.com/mchadolias/miniboone-classification/actions/workflows/test.yml/badge.svg)

A fully modular, reproducible machine-learning pipeline for distinguishing **electron neutrinos (signal)** from **muon neutrinos (background)** in the MiniBooNE particle-physics dataset.

This project provides:

- 🚀 Automated **download**, **loading**, **cleaning**, **processing**, and **splitting** of MiniBooNE PID data
- 📊 Publication-quality **physics-aware visualizations**
- 🧠 Machine-learning support and extensibility (XGBoost, future deep models)
- 🧪 A comprehensive, structured **test suite** (unit + integration + performance)
- 🛠 Modern **project architecture** following PyPA and scientific computing best practices
- 🔍 A research workflow with **data lineage**, reproducibility, and structured configuration

## 🚀 Key Features

### 🧬 Pipeline

- Unified data ingestion (Kaggle or local files)
- Robust cleaning of NaNs, MiniBooNE sentinel values, and duplicates
- Physics-aware preprocessing and feature transformations
- Flexible outputs: **NumPy arrays** or **Pandas DataFrames**

### 📊 Visualization

- Signal–background separation plots (KDE, histograms)
- Correlation analysis and feature summary plots
- PCA, t-SNE, and other embedding visualizations
- Publication-ready scientific styling (LaTeX optional)

### 📐 Statistical Toolkit

- Effect size computation (Cohen’s *d*, rank-biserial)
- Hypothesis testing and multi-comparison correction
- Feature separability scoring and ranking

### 🧪 Testing Framework

- Mocked external dependencies (Kaggle API, I/O)
- Comprehensive unit tests for loader, cleaner, processor, and plotters
- Statistical validation tests
- Integration, smoke, and performance tests

### 🏗 Engineering & Workflow

- CI/CD via GitHub Actions
- YAML-driven logging (colored console, JSON optional)
- Standardized Makefile automation
- Modular and research-oriented `src/` architecture

---

## 📁 Project Structure

```markdown
miniboone-classification/
├── data/
│   ├── external/                         # Raw third-party datasets (e.g., MiniBooNE PID CSV)
│   └── processed/                        # Cleaned + transformed datasets ready for modeling
├── notebooks/                            # Exploratory notebooks (numbered for execution order)
│   └── 01_data_exploration.ipynb
├── src/
│   ├── config/                           # Centralized configuration & presets
│   │   ├── config.py                     # Pydantic-based config management
│   │   ├── logging.yaml                  # Logging configuration
│   │   └── presets.py                    # Plotting & model presets
│   ├── data/                       
│   │   ├── data_loader.py                # Kaggle downloader & local loader
│   │   ├── data_cleaner.py               # Missing data, outlier logic, physics adjustments
│   │   ├── data_processor.py             # Feature builders, scaling pipeline, splits
│   │   └── data_handler.py               # High-level pipeline wrapper (load → clean → process)
│   │
│   ├── plotter/                    
│   │   ├── base_plotter.py               # Scientific plotting setup + LaTeX styles
│   │   ├── neutrino_plotter.py           # Physics-aware plots (signal vs background, correlations)
│   │   └── dimensionality_plotter.py     # PCA, t-SNE, embeddings
│   ├── stats/
│   │   └── statistical_analysis.py       # Statistical tests, effect sizes, corrections
│   ├── styles/
│   │   └── plot_style.py                 # Global Matplotlib/SciPlot styling
│   └── utils/
│       ├── logger.py                     # Global logger loader (YAML-driven)
│       └── paths.py                      # Project-root resolution utilities
├── tests/
│   ├── conftest.py                       # Shared fixtures for all tests
│   ├── integration/                      # Combined integration tests
│   ├── unit/                             # Unit Tests for individual module parts
│   ├── output/                           # Temporary files generated during testing
│   ├── reports/                          # Coverage & diagnostics (HTML reports)
│   └── test_smokes.py                    # Quick, fast-running smoke tests
├── logs/                                 # Log files (if enabled in logging.yaml)
├── mlruns/                               # MLflow folder housing model registry and summaries
├── tmp/                                  # Temporary scripts & scratch files
├── Makefile                              # Build, test, clean, format commands
├── pyproject.toml                        # Dependency & build configuration
├── setup.cfg                             # Linting/formatting settings
├── LICENSE
└── README.md                      
```

## 🛠 Installation

```bash
git clone https://github.com/mchadolias/miniboone-classification
cd miniboone-classification
```

### Kaggle setup

```bash
mkdir -p ~/.config/kaggle
cp kaggle.json ~/.config/kaggle/
chmod 600 ~/.config/kaggle/kaggle.json
```

### Install dependencies

```bash
uv sync    # or: pip install -r requirements.txt
```

---

## 🎯 Usage Example

### Load → Clean → Process the dataset

```python
from src.data.data_handler import MiniBooNEDataHandler

handler = MiniBooNEDataHandler()

# Run full pipeline
df_clean, splits, pipeline = handler.run()

X_train, y_train = splits["train"]
```

### Generate physics plots

```python
from src.plotter.neutrino_plotter import NeutrinoPlotter

plotter = NeutrinoPlotter()
plotter.plot_feature_separation(df_clean, features=["feature_1", "feature_5"])
```

### Dimensionality reduction

```python
from src.plotter.dimensionality_reduction_plotter import DimensionalityReductionPlotter

dr = DimensionalityReductionPlotter()
fig = dr.plot_tsne_embedding(df_clean)
```

---

## 🧪 Testing Command Sheet

```bash
make test            # full test suite
make test-dev        # fast local tests
make test-cov        # coverage
make lint            # static analysis
make format          # format with black/isort
```

---

## 📊 About the Data

The MiniBooNE detector dataset contains:

- 50 reconstructed PMT & hit-structure features
- ~93k muon-neutrino background events
- ~36k electron-neutrino signal events

This project handles the common MiniBooNE preprocessing steps:

- Replace MiniBooNE’s sentinel missing value `-999`
- Column-wise median imputation
- Feature scaling and (optional) transforms
- Train/val/test splitting with reproducible seeds

---

## 📚 Roadmap

- [x] Full data pipeline orchestration
- [x] Physics-aware plotting module
- [x] YAML logging system
- [x] Advanced statistics module
- [ ] ML training pipeline (XGBoost, tabular NN)
- [ ] Feature importance + SHAP
- [ ] MLflow experiment tracking
- [ ] Hyperparameter search
- [ ] Add real detector-inspired feature engineering

---

## 📝 License

MIT — see [LICENSE](LICENSE).

## 📅 Last Updated

*Date: 06/12/2025*

## 👤 Author

- Michalis Chadolias  
- Email: mchadolias[@]gmail.com  
- GitHub: [https://github.com/mchadolias](https://github.com/mchadolias)

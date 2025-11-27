# MiniBooNE Particle Classification

![Tests](https://github.com/mchadolias/miniboone-classification/actions/workflows/test.yml/badge.svg)

A machine learning project to distinguish electron neutrinos (signal) from muon neutrinos (background) based on reconstructed detector event features from the MiniBooNE experiment.

## 🚀 Features

- **Data Handling**: Automated download and preprocessing of MiniBooNE dataset
- **Machine Learning**: XGBoost model for particle classification  
- **Testing**: Comprehensive test suite with pytest and coverage
- **CI/CD**: Automated testing on every commit
- **Code Quality**: Linting and formatting with flake8, black, isort

## 📁 Project Structure

```markdown
miniboone-classification/
├── data
│   ├── external              # Data from third party sources.
│   └── processed             # The final, canonical data sets for modeling.
├── notebooks                 # Jupyter notebooks. Naming convention is a number (for ordering)
│   └── 01_data_exploration.ipynb 
├── src/
│   ├── config.py             # Configuration management
│   ├── data/
│   │   └── data_handler.py   # Data loading and preprocessing
│   └── visualization/
│       └── plotter.py        # Visualization utilities
├── tests/                    # Test suite
├── .github/workflows/        # CI/CD pipeline
├── pyproject.toml            # Project dependencies
├── Makefile                  # Development commands
├── figures/                   # Figures
├── models/                   # Trained and serialized models, model predictions, or model summaries
├── LICENSE                   # Open-source license
└── README.md                 # Project description
```

## 🛠️ Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/mchadolias/miniboone-classification
   cd miniboone-classification
   ```

2. **Set up Kaggle credentials**:

   ```bash
   mkdir -p ~/.config/kaggle
   # Add your kaggle.json to ~/.config/kaggle/
   chmod 600 ~/.config/kaggle/kaggle.json
   ```

3. **Install dependencies**:

   ```bash
   make requirements
   ```

## 🎯 Usage

```python
from src.data.data_handler import MiniBooNEDataHandler

handler = MiniBooNEDataHandler()
handler.download()
df = handler.load()
df = handler.clean_data()
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run tests with coverage
make test-cov

# Run quick development tests
make test-dev

# Check code quality
make lint

# Format code
make format
```

**Test Coverage**: Includes unit tests, integration tests, and mocked external API calls.

## 🔄 CI/CD

Automated testing on every commit via GitHub Actions:

- Runs test suite with coverage
- Checks code quality
- Uploads coverage reports
- Tests on multiple platforms

## 🏗️ Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run specific test groups
make test-unit
make test-integration

# Clean temporary files
make clean

# View coverage report
make open-cov
```

## 📊 Data

- **50 reconstructed detector features**
- **Signal**: Electron neutrino events (~28%)
- **Background**: Muon neutrino events
- **Automated download** from Kaggle

## ✅ TODO List

- [ ] Add detailed tests for the visualization utilities
- [ ] Implement model training pipeline
- [ ] Add XGBoost model with hyperparameter tuning
- [ ] Create model evaluation and metrics
- [ ] Add feature importance analysis
- [ ] Add visualization utilities for results
- [ ] Create comprehensive model tests
- [ ] Add experiment tracking (MLflow/Weights & Biases)
- [ ] Implement cross-validation strategies
- [ ] Add feature importance analysis

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

--------
*Last Updated: 2025-11-28*

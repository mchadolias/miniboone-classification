# MiniBooNE-Classification

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Build and deploy a machine learning model to distinguish electron neutrinos (signal) from muon neutrinos (background), based on reconstructed detector event features.

## Project Organization

```markdown
├── data
│   ├── external                    <- Data from third party sources.
│   └── processed                   <- The final, canonical data sets for modeling.
├── figures                         <- Figures
├── LICENSE                         <- Open-source license
├── Makefile                        <- Makefile with convenience commands like `make data` or `make train`
├── models                          <- Trained and serialized models, model predictions, or model summaries
├── notebooks                       <- Jupyter notebooks. Naming convention is a number (for ordering)
│   └── 01_data_exploration.ipynb
├── pyproject.toml                  <- Project configuration file with package metadata for             miniboone_classification
│                                    and configuration for tools like black
├── README.md
├── setup.cfg                       <- Configuration file for flake8
└── src                             <- Source code for use in this project.
|    ├── __init__.py
|    ├── config.py
|    ├── data
|    │   ├── __init__.py
|    │   └── data_handler.py
|    └── visualization
|        ├── __init__.py
|        └── plotter.py
├── tests
│   └── test_data.py
└── uv.lock
```

--------


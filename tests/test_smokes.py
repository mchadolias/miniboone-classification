def test_import_core_modules():
    """Smoke test: Can we import all core modules without errors?"""
    # This should not raise any ImportError
    from src.config import DataConfig, SaveConfig, ViolinPlotConfig
    from src.visualization.plotter import NeutrinoPlotter
    from src.data.data_handler import MiniBooNEDataHandler

    # If we get here, imports work ✅
    assert True


def test_import_plotting_dependencies():
    """Smoke test: Are plotting dependencies available?"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np

    # Basic functionality check
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    plt.close(fig)

    assert True  # If we get here, dependencies work


def test_config_instantiation():
    """Smoke test: Can we create config objects without crashes?"""
    from src.config import DataConfig, SaveConfig, ViolinPlotConfig

    # Just create them - don't test specific values
    data_config = DataConfig()
    save_config = SaveConfig()
    violin_config = ViolinPlotConfig()

    # If no exceptions, smoke test passes
    assert data_config is not None
    assert save_config is not None
    assert violin_config is not None


def test_config_presets_exist():
    """Smoke test: Do our preset configurations exist?"""
    from src.config import VIOLIN_PRESETS, BOXPLOT_PRESETS

    # Just check they exist and have content
    assert VIOLIN_PRESETS is not None
    assert BOXPLOT_PRESETS is not None
    assert len(VIOLIN_PRESETS) > 0
    assert len(BOXPLOT_PRESETS) > 0


def test_data_handler_initialization():
    """Smoke test: Does data handler initialize without Kaggle errors?"""
    from src.data.data_handler import MiniBooNEDataHandler

    # This should work even without Kaggle credentials
    handler = MiniBooNEDataHandler()
    assert handler is not None


def test_full_import_chain():
    """Smoke test: Can we import the entire chain without errors?"""
    # This tests the most critical import path
    from src.config import DataConfig, SaveConfig
    from src.visualization.plotter import NeutrinoPlotter
    from src.data.data_handler import MiniBooNEDataHandler

    # Create instances
    data_config = DataConfig()
    save_config = SaveConfig()
    plotter = NeutrinoPlotter()
    data_handler = MiniBooNEDataHandler()

    # If we get here, the basic ecosystem works
    assert True

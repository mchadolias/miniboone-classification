import warnings
import matplotlib.pyplot as plt
from src.utils.logger import get_global_logger

logger = get_global_logger(__name__)

try:
    import scienceplots  # noqa: F401

    SCIENCEPLOTS_AVAILABLE = True
except ImportError:
    SCIENCEPLOTS_AVAILABLE = False
    warnings.warn("scienceplots not available, using default matplotlib styles")


def check_latex_available() -> bool:
    original_usetex = plt.rcParams.get("text.usetex", False)
    try:
        plt.rcParams["text.usetex"] = True
        fig, ax = plt.subplots(figsize=(1, 1), dpi=80)
        ax.text(0.5, 0.5, r"$\\alpha$", transform=ax.transAxes, ha="center", va="center")
        fig.canvas.draw_idle()
        return True
    except Exception as e:
        logger.debug(f"LaTeX test failed: {e}")
        return False
    finally:
        plt.rcParams["text.usetex"] = original_usetex
        plt.close("all")


def setup_scientific_plotting(style: str = "science") -> None:
    original_usetex = plt.rcParams.get("text.usetex", False)
    try:
        if style == "science" and SCIENCEPLOTS_AVAILABLE:
            if check_latex_available():
                plt.style.use(["science", "ieee", "grid"])
                plt.rcParams["text.usetex"] = True
                logger.info("Using scienceplots style with LaTeX rendering")
            else:
                plt.style.use(["science", "ieee", "grid", "no-latex"])
                plt.rcParams["text.usetex"] = False
                logger.info("Using scienceplots style without LaTeX rendering")
        elif style == "seaborn":
            plt.rcParams["text.usetex"] = False
            plt.style.use("seaborn-v0_8-whitegrid")
            logger.info("Using seaborn style")
        else:
            plt.rcParams["text.usetex"] = False
            plt.style.use("default")
            logger.info("Using default matplotlib style")
    except Exception as e:
        logger.warning(f"Could not set style {style}: {e}")
        plt.style.use("default")
        plt.rcParams["text.usetex"] = original_usetex

    plt.rcParams.update(
        {
            "figure.figsize": (10, 8),
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "serif",
        }
    )

    if not plt.rcParams.get("text.usetex", False):
        plt.rcParams["mathtext.fontset"] = "stix"
        logger.info("Using STIX math font (non-LaTeX mode)")

import click

from src.config.config import DataConfig, MLConfig
from src.data.data_handler import MiniBooNEDataHandler
from src.ml.experiment import Experiment
from src.utils.logger import get_global_logger


@click.command()
@click.option("--model", "-m", default="random_forest")
@click.option("--n-estimators", "-n", default=200, type=int)
@click.option("--simple/--no-simple", default=True)
@click.option("--use-optuna", is_flag=True)
@click.option("--trials", "-t", default=30, type=int)
@click.option("--sampler", "-s", default="tpe", type=click.Choice(["tpe", "random", "cmaes"]))
@click.option("--pruning", is_flag=True)
@click.option("--timeout", default=600, type=int)
@click.option("--cv-folds", "-k", default=5, type=int)
@click.option("--experiment-name", "-e", default="miniboone_experiment")
@click.option("--run-name", "-r", default="default_run")
@click.option("--no-cache", is_flag=True)
@click.option(
    "--final-test/--no-final-test",
    default=False,
    type=bool,
    help="Whether to run evaluation on final test set",
)
def main(
    model,
    n_estimators,
    simple,
    use_optuna,
    trials,
    sampler,
    pruning,
    timeout,
    cv_folds,
    experiment_name,
    run_name,
    no_cache,
    final_test,
):
    """Main training script for MiniBooNE classification experiments."""

    # -------------------------------------------------------------
    # CONFIG OBJECTS
    # -------------------------------------------------------------
    ml_cfg = MLConfig(
        experiment_name=experiment_name,
        run_name=run_name,
        model_name=model,
        model_params={"n_estimators": n_estimators},
        simple_training=simple,
        use_optuna=use_optuna,
        n_trials=trials,
        optuna_sampler=sampler,
        optuna_pruning=pruning,
        optuna_timeout=timeout,
        cv_folds=cv_folds,
        final_test=final_test,
    )

    logger = get_global_logger(__name__)
    logger.info("Initialising training configuration...")

    data_cfg = DataConfig()

    if no_cache:
        data_cfg.use_cache = False
        logger.warning("CACHE DISABLED → full preprocessing pipeline will run.")

    # -------------------------------------------------------------
    # PRINT RESOLVED PATHS (IMPORTANT)
    # -------------------------------------------------------------
    logger.info(f"Resolved data_dir:  {data_cfg.data_dir}")
    logger.info(f"Resolved cache_dir: {data_cfg.cache_dir}")

    # -------------------------------------------------------------
    # DATA HANDLER
    # -------------------------------------------------------------
    data_handler = MiniBooNEDataHandler(config=data_cfg)

    # -------------------------------------------------------------
    # EXPERIMENT
    # -------------------------------------------------------------
    logger.info("Starting experiment...")
    experiment = Experiment(cfg=ml_cfg, data_handler=data_handler)
    result, evaluation = experiment.run()

    logger.info("Training finished!")
    logger.info(f"Test ROC-AUC: {evaluation['roc_auc']:.4f}")
    logger.info(f"Test F1 Score: {evaluation['f1_score']:.4f}")


if __name__ == "__main__":
    main()

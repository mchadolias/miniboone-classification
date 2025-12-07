import logging

import mlflow
import mlflow.sklearn
import pandas as pd

from src.ml.evaluator import evaluate_model
from src.ml.trainer import Trainer


class Experiment:
    """
    Conducts a full experiment using MLflow and DataHandler.
    Supports:
      - Cached preprocessing
      - Simple baseline training
      - Optuna hyperparameter search
    """

    def __init__(self, cfg, data_handler, logger=None):
        self.cfg = cfg
        self.data_handler = data_handler
        self.logger = logger or logging.getLogger(__name__)

    # -------------------------------------------------------------
    # Cache check
    # -------------------------------------------------------------
    def _get_cached_data(self):
        dp = self.data_handler.processor
        if self.data_handler.df is None:
            self.data_handler.get_data()

        df = self.data_handler.df
        key = dp._get_cache_key(df)
        paths = dp._get_cache_paths(key)
        valid = dp._is_cache_valid(paths)

        return valid, paths

    # -------------------------------------------------------------
    # Main experiment
    # -------------------------------------------------------------
    def run(self):
        mlflow.set_experiment(self.cfg.experiment_name)

        with mlflow.start_run(run_name=self.cfg.run_name):
            self.logger.info("────────────────────────────────────────────")
            self.logger.info(f"Experiment: {self.cfg.run_name}")
            self.logger.info("────────────────────────────────────────────")

            # Log config (top level only)
            mlflow.log_params(
                {
                    "model_name": self.cfg.model_name,
                    "simple_training": self.cfg.simple_training,
                    "use_optuna": self.cfg.use_optuna,
                    "cv_folds": self.cfg.cv_folds,
                }
            )

            # ---------------------------------------------------------
            # DATA PROCESSING (use cache if available)
            # ---------------------------------------------------------
            has_cache, cache_paths = self._get_cached_data()

            if has_cache and self.data_handler.config.use_cache:
                self.logger.info("Using cached processed dataset.")
                X_train, X_test, y_train, y_test = self.data_handler.processor._load_cache(
                    cache_paths
                )
            else:
                self.logger.info("Cache missing → full preprocessing.")
                self.data_handler.load()
                self.data_handler.clean_data()
                splits = self.data_handler.process(clean=False)

                X_train, y_train = splits["train"]
                X_test, y_test = splits["test"]

            if X_train is None or X_test is None:
                raise ValueError("Processed data is None. Cannot proceed with training.")

            if isinstance(X_train, pd.DataFrame) and isinstance(X_test, pd.DataFrame):
                self.logger.info("Processed data is in DataFrame format.")
                X_train = X_train.to_numpy()
                X_test = X_test.to_numpy()

            # ---------------------------------------------------------
            # TRAINING
            # ---------------------------------------------------------
            trainer = Trainer(self.cfg.model_name, logger=self.logger)

            if self.cfg.use_optuna:
                self.logger.info("Mode: OPTUNA optimisation")

                result = trainer.optimize(
                    X_train,
                    y_train,
                    n_trials=self.cfg.n_trials,
                    cv_folds=self.cfg.cv_folds,
                    direction=self.cfg.optuna_direction,
                    timeout=self.cfg.optuna_timeout,
                    sampler_name=self.cfg.optuna_sampler,
                    use_pruning=self.cfg.optuna_pruning,
                )

                mlflow.log_params(result.metrics["best_params"])
                mlflow.log_metric("optuna_best_score", result.metrics["best_score"])

            else:
                self.logger.info("Mode: SIMPLE TRAINING")

                result = trainer.simple_train(
                    X_train,
                    y_train,
                    model_params=self.cfg.model_params,
                    cv_folds=self.cfg.cv_folds,
                )

                if "roc_auc_mean" in result.metrics:
                    mlflow.log_metric("cv_roc_auc_mean", result.metrics["roc_auc_mean"])
                    mlflow.log_metric("cv_roc_auc_std", result.metrics["roc_auc_std"])

            final_model = result.model

            # ---------------------------------------------------------
            # EVALUATION
            # ---------------------------------------------------------
            evaluation = evaluate_model(
                final_model,
                X_test,
                y_test,
            )

            mlflow.log_metric("test_roc_auc", evaluation["roc_auc"])
            mlflow.log_metric("test_f1", evaluation["f1_score"])

            mlflow.log_dict(evaluation["classification_report"], "classification_report.json")
            mlflow.log_dict(
                {"confusion_matrix": evaluation["confusion_matrix"]}, "confusion_matrix.json"
            )

            # ---------------------------------------------------------
            # ARTIFACTS
            # ---------------------------------------------------------
            mlflow.sklearn.log_model(final_model, "model")

            try:
                pipeline_path = self.data_handler.processor.export_pipeline()
            except ValueError as e:
                # Most likely running on cached splits in a fresh process: no in-memory pipeline
                self.logger.warning("Skipping pipeline export: %s", e)
            else:
                mlflow.log_artifact(str(pipeline_path))

            self.logger.info("Experiment completed successfully.")
            return result, evaluation

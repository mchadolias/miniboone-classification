from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator

from src.ml.model_factory import create_model
import logging


@dataclass
class TrainingResult:
    model: BaseEstimator
    metrics: Dict[str, Any]
    cv_scores: Optional[np.ndarray] = None
    study: Optional[optuna.Study] = None


class Trainer:
    """
    Trainer supporting:
      - Simple baseline training (with optional CV)
      - Optuna Bayesian hyperparameter optimisation
      - Unified logging
    """

    def __init__(self, model_name: str, logger=None):
        self.model_name = model_name
        self.logger = logger or logging.getLogger(__name__)

    # =====================================================================
    # SIMPLE TRAINING MODE
    # =====================================================================
    def simple_train(self, X, y, model_params: dict, cv_folds: int = 5) -> TrainingResult:
        """
        Train a model using fixed hyperparameters (no optimisation).
        """
        self.logger.info("Running SIMPLE TRAINING for model '%s'", self.model_name)

        model = create_model(self.model_name, model_params)

        return self.fit(model, X, y, cv=cv_folds)

    # =====================================================================
    # BASE FIT FUNCTION (works for both modes)
    # =====================================================================
    def fit(self, model: BaseEstimator, X, y, cv: Optional[int] = None) -> TrainingResult:
        """
        Fit model with optional k-fold CV.
        """
        metrics = {}

        if cv and cv > 1:
            self.logger.info("Performing %d-fold cross-validation...", cv)
            scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
            metrics["roc_auc_mean"] = float(np.mean(scores))
            metrics["roc_auc_std"] = float(np.std(scores))

            self.logger.info(
                "CV ROC-AUC: mean=%.4f  std=%.4f", metrics["roc_auc_mean"], metrics["roc_auc_std"]
            )
        else:
            scores = None
            self.logger.info("Skipping CV.")

        self.logger.info("Fitting final model on full dataset...")
        model.fit(X, y)

        return TrainingResult(model=model, metrics=metrics, cv_scores=scores)

    # =====================================================================
    # OPTUNA OPTIMISATION MODE
    # =====================================================================
    def optimize(
        self,
        X,
        y,
        n_trials: int,
        cv_folds: int,
        direction: str,
        timeout: Optional[int],
        sampler_name: str,
        use_pruning: bool,
    ) -> TrainingResult:

        self.logger.info(
            f"Starting OPTUNA optimisation → model='{self.model_name}', "
            f"trials={n_trials}, cv={cv_folds}"
        )

        # --------------------
        # Choose sampler
        # --------------------
        if sampler_name == "tpe":
            sampler = optuna.samplers.TPESampler(seed=42)
        elif sampler_name == "random":
            sampler = optuna.samplers.RandomSampler(seed=42)
        elif sampler_name == "cmaes":
            sampler = optuna.samplers.CmaEsSampler(seed=42)
        else:
            raise ValueError(f"Unknown Optuna sampler '{sampler_name}'")

        # --------------------
        # Choose pruner
        # --------------------
        pruner = optuna.pruners.MedianPruner() if use_pruning else None

        study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)

        # --------------------
        # Objective Function
        # --------------------
        def objective(trial):
            params = self._suggest_params(trial)
            self.logger.debug(f"[Trial {trial.number}] Params: {params}")

            model = create_model(self.model_name, params)

            scores = cross_val_score(model, X, y, cv=cv_folds, scoring="roc_auc", n_jobs=-1)

            score_mean = float(np.mean(scores))

            self.logger.info(f"[Trial {trial.number}] ROC-AUC = {score_mean:.4f}")

            trial.report(score_mean, step=0)

            if use_pruning and trial.should_prune():
                self.logger.warning(f"[Trial {trial.number}] PRUNED")
                raise optuna.TrialPruned()

            return score_mean

        # --------------------
        # Run optimisation
        # --------------------
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        self.logger.info("──────────────────────────────────────────────")
        self.logger.info("  OPTUNA BEST SCORE:  %.5f", study.best_value)
        self.logger.info("  BEST PARAMS:        %s", study.best_params)
        self.logger.info("──────────────────────────────────────────────")

        # --------------------
        # Fit final best model
        # --------------------
        best_model = create_model(self.model_name, study.best_params)
        best_model.fit(X, y)

        return TrainingResult(
            model=best_model,
            metrics={
                "best_score": float(study.best_value),
                "best_params": study.best_params,
                "n_trials": len(study.trials),
            },
            study=study,
        )

    # =====================================================================
    # SEARCH SPACE DEFINITIONS
    # =====================================================================
    def _suggest_params(self, trial):
        """
        Define hyperparameter search space for each supported model.
        """

        if self.model_name == "logistic_regression":
            return {
                "C": trial.suggest_loguniform("C", 1e-3, 10),
                "max_iter": trial.suggest_int("max_iter", 100, 800),
                "solver": "liblinear",
            }

        if self.model_name == "random_forest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 800),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            }

        if self.model_name == "xgboost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }

        raise ValueError(f"No search space defined for model '{self.model_name}'")

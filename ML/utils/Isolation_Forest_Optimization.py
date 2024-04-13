from sklearn.ensemble import IsolationForest
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
)
import numpy as np
import pandas as pd
import typing


class IsolationForestOptimizer:
    """
    This class is used to optimize the hyperparameters of an algorithm using Isolation Forest for outlier detection.
    """

    def __init__(
        self,
        algorithm: typing.Any,
        metric: str,
        cv: typing.Any = KFold(n_splits=5, shuffle=True, random_state=17),
        n_trials: int = 100,
        seed: int = 17,
    ) -> None:
        """
        Initializes the IsolationForestOptimizer class.

        Args:
            algorithm (typing.Any): algorithm to optimize.
            metric (str): metric to use for optimization.
            cv (typing.Any): cross-validation strategy.
            n_trials (int): number of trials to perform.
            seed (int): random seed.
        """
        self.algorithm = algorithm
        metrics = {
            "accuracy": [lambda y, y_pred: accuracy_score(y, y_pred), "preds"],
            "roc_auc": [lambda y, y_pred: roc_auc_score(y, y_pred), "probs"],
            "mse": [lambda y, y_pred: mean_squared_error(y, y_pred), "preds"],
            "rmse": [
                lambda y, y_pred: mean_squared_error(y, y_pred) ** 0.5,
                "preds",
            ],
            "mae": [lambda y, y_pred: mean_absolute_error(y, y_pred), "preds"],
        }
        if metric not in metrics:
            raise ValueError("Unsupported metric: {}".format(metric))
        self.metric = metric
        self.eval_metric = metrics[metric][0]
        self.metric_type = metrics[metric][1]
        self.cv = cv
        self.n_trials = n_trials
        self.seed = seed

    def optimize(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> typing.Tuple[float, float, float, dict]:
        """
        This method optimizes the hyperparameters of the algorithm using Isolation Forest for outlier detection.

        Args:
            X_train (pd.DataFrame): training data.
            y_train (pd.Series): training labels.
            X_test (pd.DataFrame): testing data.
            y_test (pd.Series): testing labels.

        Returns:
            typing.Tuple[float, float, float, dict]: ratio of outliers, cross-validation scores, test score, and best hyperparameters.
        """
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train),
            n_trials=self.n_trials,
        )
        isolation_forest_best_params = study.best_params
        return self.evaluate(
            X_train, y_train, X_test, y_test, isolation_forest_best_params
        )

    def objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        """
        This method defines the objective function for optimization.

        Args:
            trial (optuna.Trial): trial object.
            X (pd.DataFrame): input data.
            y (pd.Series): target data.

        Returns:
            float: cross-validation score.
        """
        isolation_forest_params = {
            "n_estimators": trial.suggest_int("n_estimators", low=50, high=200),
            "contamination": trial.suggest_float("contamination", low=0, high=0.1),
            "bootstrap": trial.suggest_categorical("bootstrap", [True]),  # constant
            "n_jobs": trial.suggest_categorical("n_jobs", [-1]),  # constant
            "random_state": trial.suggest_categorical(
                "random_state", [self.seed]
            ),  # constant
            "verbose": 0,
        }
        isolation_forest = IsolationForest(**isolation_forest_params)
        scores = []
        for train_idx, valid_idx in self.cv.split(X):
            X_train_cv, X_valid_cv = X.iloc[train_idx, :], X.iloc[valid_idx, :]
            y_train_cv, y_valid_cv = y.iloc[train_idx], y.iloc[valid_idx]
            isolation_forest.fit(X_train_cv)
            X_train_no_outliers = X_train_cv.loc[
                isolation_forest.predict(X_train_cv) == 1
            ]
            y_train_no_outliers = y_train_cv.loc[
                isolation_forest.predict(X_train_cv) == 1
            ]
            self.algorithm.fit(X_train_no_outliers, y_train_no_outliers)
            if self.metric_type == "preds":
                y_pred = self.algorithm.predict(X_valid_cv)
            else:
                y_pred = self.algorithm.predict_proba(X_valid_cv)[:, 1]
            y_pred = self.algorithm.predict(X_valid_cv)
            scores.append(self.eval_metric(y_valid_cv, y_pred))
        return np.mean(scores)

    def evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        isolation_forest_best_params: dict,
    ) -> typing.Tuple[typing.List[int], float, float, float, dict]:
        """
        This method evaluates the optimized algorithm.

        Args:
            X_train (pd.DataFrame): training data.
            y_train (pd.Series): training labels.
            X_test (pd.DataFrame): testing data.
            y_test (pd.Series): testing labels.
            isolation_forest_best_params (dict): best hyperparameters.

        Returns:
            typing.Tuple[typing.List[int], float, float, float, dict]: outliers indices, ratio of outliers, cross-validation scores, test score, and best hyperparameters.
        """
        isolation_forest = IsolationForest(**isolation_forest_best_params)
        isolation_forest.fit(X_train)
        X_train_no_outliers = X_train.loc[isolation_forest.predict(X_train) == 1]
        y_train_no_outliers = y_train.loc[isolation_forest.predict(X_train) == 1]
        outliers_indices = list(
            X_train.loc[isolation_forest.predict(X_train) == -1].index
        )
        ratio_of_outliers = (
            X_train.shape[0] - X_train_no_outliers.shape[0]
        ) / X_train.shape[0]
        cv_scores = self.perform_cv(X_train_no_outliers, y_train_no_outliers)
        self.algorithm.fit(X_train_no_outliers, y_train_no_outliers)
        if self.metric_type == "preds":
            y_pred = self.algorithm.predict(X_test)
        else:
            y_pred = self.algorithm.predict_proba(X_test)[:, 1]
        test_score = self.eval_metric(y_test, y_pred)
        return (
            outliers_indices,
            ratio_of_outliers,
            cv_scores,
            test_score,
            isolation_forest_best_params,
        )

    def perform_cv(self, X: pd.DataFrame, y: pd.Series) -> float:
        """This method performs cross-validation.

        Args:
            X: (pd.DataFrame): input data.
            y: (pd.Series): target data.

        Returns:
            float: cross-validation score.
        """
        scores = []
        for train_idx, valid_idx in self.cv.split(X):
            X_train_cv, X_valid_cv = X.iloc[train_idx, :], X.iloc[valid_idx, :]
            y_train_cv, y_valid_cv = y.iloc[train_idx], y.iloc[valid_idx]
            self.algorithm.fit(X_train_cv, y_train_cv)
            if self.metric_type == "preds":
                y_pred = self.algorithm.predict(X_valid_cv)
            else:
                y_pred = self.algorithm.predict_proba(X_valid_cv)[:, 1]
            scores.append(self.eval_metric(y_valid_cv, y_pred))
        return np.mean(scores)

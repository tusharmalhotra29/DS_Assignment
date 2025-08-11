from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, roc_auc_score


class Classifier(ABC):
    @abstractmethod
    def train(self, *params) -> None:
        pass

    @abstractmethod
    def evaluate(self, *params) -> Dict[str, float]:
        pass

    @abstractmethod
    def predict(self, *params) -> np.ndarray:
        pass


class SklearnClassifier(Classifier):
    def __init__(
        self, estimator: BaseEstimator, features: List[str], target: str,
    ):
        self.clf = estimator
        self.features = features
        self.target = target

    def train(self, df_train: pd.DataFrame):
        self.clf.fit(df_train[self.features].values, df_train[self.target].values)

    def evaluate(self, df_test: pd.DataFrame):
        X_test = df_test[self.features]
        y_test = df_test[self.target]

        y_pred = self.clf.predict(X_test)
        y_proba = self.clf.predict_proba(X_test)[:, 1]  # probability of positive class

        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        return {
            "f1_score": f1,
            "roc_auc": roc_auc
        }

    def predict(self, df: pd.DataFrame):
        return self.clf.predict_proba(df[self.features].values)[:, 1]

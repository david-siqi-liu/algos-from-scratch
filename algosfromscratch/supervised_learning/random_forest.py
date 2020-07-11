from collections import namedtuple
from typing import Any
from math import log

import numpy as np

from algosfromscratch.supervised_learning import DecisionTreeClassifier
from algosfromscratch.utils.tree import get_unique_values, get_majority_class

Estimator = namedtuple('Estimator', ('tree', 'samples', 'features'))


class RandomForestClassifier:

    def __init__(self, n_estimators: int = 100, criterion: str = 'gini', max_depth: int = None,
                 min_samples_split: int = 2, max_features: Any = 'auto', min_impurity_decrease: float = 0.0,
                 bootstrap: bool = True, random_state: int = None, max_samples: Any = None):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth if max_depth else float('inf')
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.random_instance = np.random.mtrand._rand if not random_state else np.random.RandomState(random_state)
        self.max_samples = max_samples

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Build a random forest classifier from the training set (X, y).
        """
        # Shape and number of classes
        self.n_samples_, self.n_features_ = X.shape
        self.n_classes_ = len(get_unique_values(y))
        # Initialize an array to store the fitted estimators
        self.estimators_ = []
        # Build n_estimators number of DecisionTreeClassifier
        for i in range(self.n_estimators):
            # Randomly select samples (SRSWOR)
            if not self.bootstrap or not self.max_samples:
                samples_i = np.arange(self.n_samples_)
            elif isinstance(self.max_samples, int):
                samples_i = self.random_instance.randint(0, self.n_samples_,
                                                         max(2, min(self.n_samples_, self.max_samples)))
            elif isinstance(self.max_samples, float):
                samples_i = self.random_instance.randint(0, self.n_samples_,
                                                         max(2, int(min(1.0, self.max_samples) * self.n_samples_)))
            else:
                samples_i = np.arange(self.n_samples_)
            # Randomly select features (SRSWOR)
            if self.max_features is None:
                n_features_i = self.max_features
            elif isinstance(self.max_features, int):
                n_features_i = self.max_features
            elif isinstance(self.max_features, float):
                n_features_i = int(self.max_features * self.n_features_)
            elif self.max_features in ['auto', 'sqrt']:
                n_features_i = int(np.sqrt(self.n_features_))
            elif self.max_features == 'log2':
                n_features_i = int(log(self.n_features_, 2))
            else:
                n_features_i = self.max_features
            features_i = self.random_instance.randint(0, self.n_features_,
                                                      max(1, min(self.n_features_, n_features_i)))
            # Initialize a DecisionTreeClassifier
            tree_i = DecisionTreeClassifier(criterion=self.criterion,
                                            max_depth=self.max_depth,
                                            min_samples_split=self.min_samples_split,
                                            min_impurity_decrease=self.min_impurity_decrease)
            # Fit using the randomly selected sample and feature subsets
            tree_i.fit(X[samples_i, :][:, features_i], y[samples_i])
            # Store in array
            self.estimators_.append(Estimator(tree_i, samples_i, features_i))

    def classify(self, x: np.ndarray) -> Any:
        """
        Classifies a single sample x using majority rule.
        """
        y_preds = np.array([i.tree.classify(x[i.features], i.tree.root) for i in self.estimators_])
        return get_majority_class(y_preds)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict classes for X.
        """
        if not self.estimators_:
            raise SystemError('Tree has not been fitted yet.')
        return np.array([self.classify(x) for x in X])

from collections import defaultdict
from math import log

import numpy as np

from algosfromscratch.supervised_learning import DecisionTreeClassifier
from algosfromscratch.utils.tree import get_unique_values


class AdaBoostClassifier:

    def __init__(self, n_estimators=50, learning_rate=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate  # shrinks the contribution of each classifier
        if not random_state:
            self.random_instance = np.random.mtrand._rand
        elif isinstance(random_state, int):
            self.random_instance = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.random_instance = random_state

    def fit(self, X, y):
        """
        Build a AdaBoost classifier from the training set (X, y).
        """
        # Shape and number of classes
        self.n_samples_, self.n_features_ = X.shape
        self.classes_ = get_unique_values(y)
        self.n_classes_ = len(self.classes_)
        # Initialize an array to store the fitted estimators
        self.estimators_ = []
        # Initialize an array to store the weights for each estimators in the boosted ensemble
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        # Initialize an array to store the classification error for each estimator in the boosted ensemble
        self.estimator_errors_ = np.zeros(self.n_estimators, dtype=np.float64)
        # Initialize sample weights (equal to all)
        sample_weights = np.ones(self.n_samples_) / self.n_samples_
        # Build n_estimators number of DecisionTreeClassifier stump trees
        for i in range(self.n_estimators):
            # SRSWR using the sample weights
            samples_i = self.random_instance.choice(a=np.arange(self.n_samples_),
                                                    size=self.n_samples_,
                                                    replace=True,
                                                    p=sample_weights)
            # Initialize a DecisionTreeClassifier
            tree_i = DecisionTreeClassifier(max_depth=1, random_state=self.random_instance)
            # Fit using the randomly selected sample and feature subsets
            tree_i.fit(X[samples_i, :], y[samples_i])
            # Store in array
            self.estimators_.append(tree_i)
            # Total error (weighted by sample weights)
            y_preds_i = tree_i.predict(X)
            incorrect_i = y_preds_i != y
            total_error_i = np.sum(np.dot(sample_weights, incorrect_i))
            self.estimator_errors_[i] = total_error_i
            # No error
            if total_error_i == 0.0:
                sig_lvl_i = self.learning_rate * log((self.n_classes_ - 1) / np.finfo(float).eps)
            else:
                sig_lvl_i = self.learning_rate * log((self.n_classes_ - 1) * (1 - total_error_i) / total_error_i)
            # Store in array
            self.estimator_weights_[i] = sig_lvl_i
            # Early termination if no error
            if total_error_i == 0.0:
                break
            # Update weights for the next iteration
            # If incorrect, *= e^(significance level)
            # If correct, unchanged
            sample_weights *= np.exp(sig_lvl_i * incorrect_i)
            # Adjust so sum equals to 1
            sample_weights /= np.sum(sample_weights)

    def classify(self, x):
        """
        Classifies a single sample x using majority rule.
        """
        votes = defaultdict(float)
        for i in range(len(self.estimators_)):
            tree_i, weight_i = self.estimators_[i], self.estimator_weights_[i]
            y_pred_i = tree_i.classify(x, tree_i.root)
            votes[y_pred_i] += weight_i
        return max(votes, key=votes.get)

    def predict(self, X):
        """
        Predict classes for X.
        """
        if not self.estimators_:
            raise SystemError('Classifier has not been fitted yet.')
        return np.array([self.classify(x) for x in X])

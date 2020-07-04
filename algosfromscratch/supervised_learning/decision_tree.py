import operator
from typing import Any

import numpy as np

from algosfromscratch.utils.tree import TreeNode, calculate_impurity, get_unique_values, get_majority_class


class DecisionTreeClassifier:
    """
    Decision Tree - Classification
    """

    def __init__(self, criterion: str = 'gini', max_depth: int = None, min_samples_split: int = 2,
                 min_impurity_decrease: float = 0.0):
        self.criterion = criterion
        self.max_depth = max_depth if max_depth else float('inf')
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Build a decision tree classifier from the training set (X, y).
        """
        self.n_samples_total = X.shape[0]
        # Prepare the features
        self._prepare_features(X)
        # Call the recursive function to build tree
        self.root = self._build_tree(X, y, 1)

    def _prepare_features(self, X: np.ndarray):
        """
        Determine each feature's type - categorical or numerical.
        """
        # Determine feature types and unique values
        self.feature_types = {}
        for feature_i in range(X.shape[1]):
            # All unique, non-NULL values
            unique_feature_values = get_unique_values(X[:, feature_i])
            # Some values are strings
            if any([isinstance(val, str) for val in unique_feature_values]):
                self.feature_types[feature_i] = 'categorical'
            # All are numbers
            else:
                self.feature_types[feature_i] = 'numerical'

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int):
        """
        Recursively builds each node in the tree.
        """
        n_samples, n_features = X.shape
        # Calculate current impurity using the given criterion
        current_impurity = calculate_impurity(y, self.criterion)
        # Calculate current majority class
        current_majority_class = get_majority_class(y)
        # Initialize return TreeNode
        current_node = TreeNode(depth=depth,
                                impurity=current_impurity,
                                majority_class=current_majority_class)
        # Stop here (i.e., make current node a leaf) if
        #   - Exceeds maximum depth
        #   - Not enough samples
        #   - Pure/impurity below threshold
        if (depth > self.max_depth) or (n_samples < self.min_samples_split) or \
                (current_impurity == 0) or (current_impurity < self.min_impurity_decrease):
            current_node.node_type = 'leaf'
            return current_node
        # Initialize trackers
        feature_best, val_best, impurity_best = None, None, float('inf')
        X1_best, X2_best, y1_best, y2_best = None, None, None, None
        # Iterate through each feature
        for feature_i in range(n_features):
            # All unique, non-NULL values
            unique_feature_values = get_unique_values(X[:, feature_i])
            # If numerical, find mid-points
            if self.feature_types[feature_i] == 'numerical' and len(unique_feature_values) > 1:
                unique_feature_values.sort()
                unique_feature_values = [(unique_feature_values[i] + unique_feature_values[i + 1]) / 2
                                         for i in range(len(unique_feature_values) - 1)]
            # Iterate through each unique value in the current feature
            for val in unique_feature_values:
                # Split using the split function
                if self.feature_types[feature_i] == 'categorical':
                    indices = (X[:, feature_i] == val)
                else:
                    indices = (X[:, feature_i] <= val)
                X1, X2, y1, y2 = X[indices], X[~indices], y[indices], y[~indices]
                # Calculate combined impurity
                impurity_after_split = (len(y1) / n_samples) * calculate_impurity(y1, self.criterion) + \
                                       (len(y2) / n_samples) * calculate_impurity(y2, self.criterion)
                # Check if it's better (i.e., lower) than the current best impurity
                if impurity_after_split < impurity_best:
                    feature_best, val_best, impurity_best = feature_i, val, impurity_after_split
                    X1_best, X2_best, y1_best, y2_best = X1, X2, y1, y2
        # Calculate weighted impurity decrease (i.e., information gain)
        weighted_impurity_decrease = (n_samples / self.n_samples_total) * (current_impurity - impurity_best)
        # If information gain is below the threshold, make current node a leaf and stop
        if (weighted_impurity_decrease == 0) or (weighted_impurity_decrease < self.min_impurity_decrease):
            current_node.node_type = 'leaf'
        # Make current node an internal node and continue expanding left and right
        else:
            current_node.node_type = 'internal'
            current_node.feature_i = feature_best
            current_node.feature_type = self.feature_types[feature_best]
            current_node.feature_val = val_best
            # Recursively build left and right trees
            current_node.left_node = self._build_tree(X1_best, y1_best, depth + 1)
            current_node.right_node = self._build_tree(X2_best, y2_best, depth + 1)
        # Return TreeNode
        return current_node

    def _classify(self, x: np.ndarray, node: TreeNode) -> Any:
        """
        Classifies a single sample x.
        """
        if node.node_type == 'leaf':
            return node.majority_class
        op = operator.eq if node.feature_type == 'categorical' else operator.le
        if op(x[node.feature_i], node.feature_val):
            return self._classify(x, node.left_node)
        else:
            return self._classify(x, node.right_node)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict classes for X.
        """
        if not self.root:
            raise SystemError('Tree has not been fitted yet.')
        return np.array([self._classify(x, self.root) for x in X])

from collections import Counter
from typing import List, Any

import numpy as np
import pandas as pd


class TreeNode:
    """
    Class that represents a node in a decision tree
    """

    def __init__(self, depth: int = None, node_type: str = None, impurity: float = None, feature_i: int = None,
                 feature_type: str = None, feature_val: Any = None, majority_class: Any = None,
                 left_node: 'TreeNode' = None, right_node: 'TreeNode' = None):
        self.depth = depth
        self.node_type = node_type  # Either 'internal' or 'leaf'
        self.impurity = impurity
        self.feature_i = feature_i  # Not applicable to leaf nodes
        self.feature_type = feature_type  # Not applicable to leaf nodes. Either 'categorical' or 'numerical'
        self.feature_val = feature_val  # Not applicable to leaf nodes
        self.majority_class = majority_class
        self.left_node = left_node  # Not applicable to leaf nodes
        self.right_node = right_node  # Not applicable to leaf nodes

    def __str__(self):
        return "<TreeNode>: depth = {depth}, node_type = {node_type}, impurity = {impurity}, feature = {feature}," \
               "feature_type = {feature_type}, feature_val = {feature_val}, majority_class = {majority_class}".format(
            depth=self.depth, node_type=self.node_type, impurity=self.impurity, feature=self.feature_i,
            feature_type=self.feature_type, feature_val=self.feature_val, majority_class=self.majority_class
        )


def export_text(node: TreeNode, feature_names: List[str]) -> str:
    """
    Recursively export the tree.
    """
    if not node:
        return ""
    return_str = ""
    prefix = "|   " * (node.depth - 1) + "|--- "
    if node.node_type == 'leaf':
        return_str += prefix
        return_str += "class: {0}\n".format(node.majority_class)
    elif node.node_type == 'internal':
        separators = ('==', '!=') if node.feature_type == 'categorical' else ('<=', '>')
        return_str += prefix
        return_str += " {0} {1} {2}\n".format(feature_names[node.feature_i], separators[0], node.feature_val)
        return_str += export_text(node.left_node, feature_names)
        return_str += prefix
        return_str += " {0} {1} {2}\n".format(feature_names[node.feature_i], separators[1], node.feature_val)
        return_str += export_text(node.right_node, feature_names)
    return return_str


def calculate_impurity(y: np.ndarray, criterion: str = 'gini') -> float:
    """
    Calculates impurity based on given criterion
    """
    n_samples = y.shape[0]
    counter = Counter(y)
    impurity = 0.0
    for v in counter.values():
        p = v / n_samples
        if criterion == 'gini':
            impurity += p * (1 - p)
        elif criterion == 'entropy':
            impurity -= p * np.log2(p)
        else:
            raise ValueError('criterion must be either gini or entropy. {0} given.'.format(criterion))
    return impurity


def get_unique_values(values: np.ndarray) -> np.ndarray:
    """
    Get all unique, non-NULL values in a ndarray.
    """
    return np.unique(values[~pd.isnull(values)])


def get_majority_class(y: np.ndarray) -> Any:
    """
    Get the most common occurring value in an array.
    """
    counter = Counter(y)
    return counter.most_common(1)[0][0]

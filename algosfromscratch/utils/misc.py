from typing import List

import numpy as np


def normalize(l: List[float]) -> List[float]:
    """
    Normalize given list to be a probability distribution.

    Args:
        l: List of floats

    Returns:
        List of floats, add up to 1
    """
    min_l = min(l)
    max_l = max(l)
    return [(i - min_l) / (max_l - min_l) for i in l]


def sigmoid(z: float) -> float:
    """
    Sigmoid activation function.

    Args:
        z: float, logit

    Returns:
        float
    """
    # To prevent overflow
    z = np.clip(z, -500, 500)
    return 1.0 / (1 + np.exp(-z))

import numpy as np


class QuadraticLoss:
    def __call__(self, y, y_pred):
        return 0.5 * np.power(y - y_pred, 2)

    def gradient(self, y, y_pred):
        return - (y - y_pred)


class CrossEntropyLoss:
    def __call__(self, y, y_pred):
        return -(np.multiply(y, np.log(y_pred)) + np.multiply(1 - y, np.log(1 - y_pred)))

    def gradient(self, y, y_pred):
        return -(np.divide(y, y_pred) - np.divide(1 - y, 1 - y_pred))

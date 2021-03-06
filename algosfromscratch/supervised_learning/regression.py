import numpy as np
from sklearn.metrics import accuracy_score

from algosfromscratch.utils.misc import sigmoid


class LinearRegression:
    """
    Linear Regression
    """

    def __init__(self, method: str = 'OLS', batch_size: int = float('inf'), learning_rate: float = 0.001,
                 num_epochs: int = 1):
        """
        Linear Regression using either Ordinary Lease Squares (OLS) or Gradient Descent (GD).

        Args:
            method: Either 'OLS' or 'GD'. Defaults to 'OLS'
            batch_size: Number of samples per batch. Applies to GD only. Defaults to inf (i.e., batch gradient descent)
            learning_rate: Learning rate. Applies to GD only. Defaults to 0.001.
            num_epochs: Number of epochs. Applies to GD only. Defaults to 1.
        """
        if method not in ['OLS', 'GD']:
            raise ValueError('Method must be either OLS or GD. {0} given.'.format(method))
        self.method = method
        if method == 'GD':
            if batch_size < 1:
                raise ValueError('Batch size must be at least 1. {0} given.'.format(batch_size))
            if learning_rate <= 0:
                raise ValueError('Learning rate must be greater than 0. {0} given.'.format(learning_rate))
            if num_epochs < 1:
                raise ValueError('Number of epochs must be at least 1. {0} given.'.format(num_epochs))
            self.batch_size = batch_size
            self.learning_rate = learning_rate
            self.num_epochs = num_epochs
            self.training_losses = []

    def _update_weights(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Update weight matrix based on the current batch using mean squared error (MSE) as loss function.

        Args:
            w: Current weights of shape (n_features, )
            X: ndarray of shape (batch_size, n_features)
            y: ndarray of shape (batch_size, )

        Returns:
            ndarray of size (1, ), MSE for current batch
        """
        # Predict using current weight
        y_pred = X.dot(w)
        # Training loss for current batch (MSE)
        loss = np.mean((y - y_pred) ** 2)
        # Gradients (by convention, we don't multiply by 2)
        w_grad = (y_pred - y).dot(X) / len(X)
        # If prediction > truth, w_grad is positive, weights decrease
        w -= self.learning_rate * w_grad
        # Return loss
        return loss

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit linear model, using either OLS or GD.

        Args:
            X: ndarray of shape (n_samples, n_features)
            y: ndarray of shape (n_samples,)

        Returns:
            None
        """
        # Prepend a column of 1's to fit the intercept
        X = np.insert(X, 0, 1, axis=1)
        if self.method == 'OLS':
            # Moore-Penrose pseudo-inverse
            # X . B = y => B = X^-1 . y
            self.intercept_ = np.linalg.pinv(X).dot(y)[0]
            self.coef_ = np.linalg.pinv(X).dot(y)[1:]
        elif self.method == 'GD':
            # Initialize weights to zeros
            w = np.zeros(X.shape[1])
            # Reset training losses
            self.training_losses = []
            # Compute number of batches
            batch_size = min(self.batch_size, len(X))
            num_batches = (len(X) - 1) // batch_size + 1
            # Train each epoch
            for epoch in range(self.num_epochs):
                # Initialize losses per batch for current epoch
                training_loss_epoch = []
                # Iterate through each batch
                batch_start = 0
                for b in range(num_batches):
                    # Batch
                    batch_end = min(len(X), batch_start + batch_size)
                    X_batch, y_batch = X[batch_start:batch_end], y[batch_start:batch_end]
                    # Update weights, return loss
                    loss = self._update_weights(w, X_batch, y_batch)
                    training_loss_epoch.append(loss)
                    # Next batch
                    batch_start = batch_end
                # Average loss for this epoch across all batches
                self.training_losses.append(np.mean(training_loss_epoch))
            # Assign intercept_ and coef_
            self.intercept_ = w[0]
            self.coef_ = w[1:]

    def predict(self, X: np.ndarray):
        """
        Predict using the linear model.

        Args:
            X: ndarray of shape (n_samples, n_features)

        Returns:
            ndarray of shape (n_samples,), predictions
        """
        assert X.shape[1] == len(self.coef_)
        X = np.insert(X, 0, 1, axis=1)
        return X.dot(np.insert(self.coef_, 0, self.intercept_))


class LogisticRegression:
    """
    Logistic Regression using Gradient Descent
    """

    def __init__(self, batch_size: int = 1, learning_rate: float = 0.0001, max_iter: int = 1, alpha: float = 0.0001):
        """
        Logistic Regression using Gradient Descent (GD) and L2 regularization.

        Args:
            batch_size: Number of samples per batch. Defaults to 1 (i.e., stochastic gradient descent).
            learning_rate: Learning rate. Applies to GD only. Defaults to 0.0001.
            max_iter: Number of epochs. Applies to GD only. Defaults to 1.
            alpha: Regularization strength. Defaults to 0.0001.
        """
        if batch_size < 1:
            raise ValueError('Batch size must be at least 1. {0} given.'.format(batch_size))
        if learning_rate <= 0:
            raise ValueError('Learning rate must be greater than 0. {0} given.'.format(learning_rate))
        if max_iter < 1:
            raise ValueError('Number of epochs must be at least 1. {0} given.'.format(max_iter))
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.training_losses = []
        self.alpha = alpha

    def _update_weights(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Update weight matrix based on the current batch using average cross-entropy as loss function.

        Args:
            w: Current weights of shape (n_features, )
            X: ndarray of shape (batch_size, n_features)
            y: ndarray of shape (batch_size, )

        Returns:
            ndarray of size (1, ), average cross-entropy loss for current batch
        """
        # Predict probability using current weight
        z = X.dot(w)
        y_pred_proba = sigmoid(z)
        # Training loss for current batch
        # Loss = - log(y_pred_proba) if y = 1;
        #        - log(1 - y_pred_proba) if y = 0
        # https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/readings/L04%20Training%20a%20Classifier.pdf
        losses = y * np.logaddexp(0, -z) + (1 - y) * np.logaddexp(0, z) + 0.5 * self.alpha * w.T.dot(w)  # L2 reg
        # Gradients, with L2 reg
        w_grad = (y_pred_proba - y).dot(X) / len(X) + self.alpha * w
        # If prediction > truth, w_grad is positive, weights decrease
        w -= self.learning_rate * w_grad
        # Return average loss
        return np.mean(losses)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit logistic model using GD.

        Args:
            X: ndarray of shape (n_samples, n_features)
            y: ndarray of shape (n_samples,)

        Returns:
            None
        """
        # Prepend a column of 1's to fit the intercept
        X = np.insert(X, 0, 1, axis=1)
        # Initialize weights to zeros
        w = np.zeros(X.shape[1])
        # Reset training losses
        self.training_losses = []
        # Compute number of batches
        batch_size = min(self.batch_size, len(X))
        num_batches = (len(X) - 1) // batch_size + 1
        # Train each epoch
        for epoch in range(self.max_iter):
            # Initialize losses per batch for current epoch
            training_loss_epoch = []
            # Iterate through each batch
            batch_start = 0
            for b in range(num_batches):
                # Batch
                batch_end = min(len(X), batch_start + batch_size)
                X_batch, y_batch = X[batch_start:batch_end], y[batch_start:batch_end]
                # Update weights, return loss
                loss = self._update_weights(w, X_batch, y_batch)
                training_loss_epoch.append(loss)
                # Next batch
                batch_start = batch_end
            # Average loss for this epoch across all batches
            self.training_losses.append(np.mean(training_loss_epoch))
        # Assign intercept_ and coef_
        self.intercept_ = w[0]
        self.coef_ = w[1:]

    def _classify(self, prob: float) -> int:
        return 1 if prob >= 0.5 else 0

    def predict(self, X: np.ndarray):
        """
        Predict using the logistic model.

        Args:
            X: ndarray of shape (n_samples, n_features)

        Returns:
            ndarray of shape (n_samples,), predictions
        """
        assert X.shape[1] == len(self.coef_)
        X = np.insert(X, 0, 1, axis=1)
        z = X.dot(np.insert(self.coef_, 0, self.intercept_))
        classify = np.vectorize(self._classify)
        return classify(sigmoid(z)).flatten()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))

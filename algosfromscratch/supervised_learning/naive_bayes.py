from collections import defaultdict, namedtuple

import numpy as np
from scipy.stats import norm

from algosfromscratch.utils.misc import normalize

Norm = namedtuple("Norm", ("mean", "std"))


class GaussianNB:
    """
    Gaussian Naive Bayes
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit Gaussian Naive Bayes according to X, y

        Args:
            X: ndarray of shape (n_samples, n_features)
            y: ndarray of shape (n_samples,)

        Returns:
            None
        """
        self.X, self.y = X, y
        self.classes = np.unique(y)
        self.num_features = self.X.shape[1]
        self.distributions = defaultdict(list)  # Dictionary of {class : List[Norm]}
        self.priors = defaultdict(float)  # Dictionary of {class : float}
        # Iterate through each class
        for c in self.classes:
            # Select samples that belong to current class
            X_in_c = self.X[self.y == c]
            # Calculate prior
            self.priors[c] = len(X_in_c) / len(self.y)
            # Calculate distributions (mean and standard deviation) for each feature
            self.distributions[c] = [Norm(np.mean(X_in_c[:, i]), np.std(X_in_c[:, i]))
                                     for i in range(self.num_features)]

    def _classify(self, X: np.ndarray) -> int:
        """
        Perform classification on one vector X.
        Pr(c | X) = Pr(X | c) * Pr(c) / Pr(X)
        posterior = likelihood * prior / evidence

        Args:
            X: ndarray of shape (n_features,)

        Returns:
            int
        """
        assert len(X) == self.num_features
        posteriors = []
        for c in self.classes:
            prior = self.priors[c]  # Pr(c)
            likelihood = 1
            for i, dist in enumerate(self.distributions[c]):
                likelihood *= norm(dist.mean, dist.std).pdf(X[i])  # Pr(X | c)
            posteriors.append(prior * likelihood)
        # Normalize to add up to 1
        posteriors = normalize(posteriors)
        # Return class with highest posterior
        return self.classes[np.argmax(posteriors)]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Perform classification on an array of vectors X.

        Args:
            X: ndarray of shape (n_samples, n_features)

        Returns:
            ndarray of shape (n_samples,)
        """
        return np.array([self._classify(x) for x in X])


class MultinomialNB:
    """
    Multinomial Naive Bayes
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit Multinomial Naive Bayes according to X, y

        Args:
            X: ndarray of shape (n_samples, n_features)
            y: ndarray of shape (n_samples,)

        Returns:
            None
        """
        self.X, self.y = X, y
        self.classes = np.unique(y)
        self.num_features = self.X.shape[1]
        self.coefs = defaultdict(list)  # Dictionary of {class : List[float]}
        self.priors = defaultdict(float)  # Dictionary of {class : float}
        # Iterate through each class
        for c in self.classes:
            # Select samples that belong to current class
            X_in_c = self.X[self.y == c]
            # Calculate prior
            self.priors[c] = len(X_in_c) / len(self.y)
            # Calculate coefficients for each feature
            self.coefs[c] = [np.sum(X_in_c[:, i]) / np.sum(X_in_c) for i in range(self.num_features)]

    def _classify(self, X: np.ndarray) -> int:
        """
        Perform classification on one vector X.
        Pr(c | X) = Pr(X | c) * Pr(c) / Pr(X)
        posterior = likelihood * prior / evidence

        Args:
            X: ndarray of shape (n_features,)

        Returns:
            int
        """
        assert len(X) == self.num_features
        # Since probabilities are too small, we will use log likelihoods instead
        log_posteriors = []
        for c in self.classes:
            log_prior = np.log(self.priors[c])  # Pr(c)
            log_likelihood = 0
            for coef, i in zip(self.coefs[c], X):
                log_likelihood += np.log(coef ** i)  # Pr(X | c)
            log_posteriors.append(log_prior + log_likelihood)
        # No need to normalize
        # Return class with highest (log) posterior
        return self.classes[np.argmax(log_posteriors)]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Perform classification on an array of vectors X.

        Args:
            X: ndarray of shape (n_samples, n_features)

        Returns:
            ndarray of shape (n_samples,)
        """
        return np.array([self._classify(x) for x in X])

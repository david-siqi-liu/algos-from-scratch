import numpy as np


class Layer(object):

    def type(self):
        return self.__class__.__name__


class LinearLayer(Layer):

    def __init__(self, name, weights):
        self.name = name
        self.W, self.b = weights
        self.n_in, self.n_out = self.W.shape

    def name(self):
        return self.name

    def num_parameters(self):
        return self.n_in * self.n_out + self.n_out

    def shape(self):
        return self.W.shape, self.b.shape

    def forward(self, A, training=True):
        """Forward pass
        """
        self.A = A
        # (n_samples, n_in) * (n_in, n_out) + (1, n_out) = (n_samples, n_out), b is broadcasted across all samples
        return np.dot(A, self.W) + self.b

    def backward(self, dZ, learning_rate):
        """Backward pass
        """
        n_samples = dZ.shape[0]
        # (n_in, n_samples) * (n_samples, n_out) = (n_in, n_out)
        dW = np.dot(self.A.T, dZ) / n_samples
        # (1, n_out)
        db = np.sum(dZ, axis=0, keepdims=True) / n_samples
        # (n_samples, n_out) * (n_out, n_in) = (n_samples, n_in)
        assert dW.shape == self.W.shape
        assert db.shape == self.b.shape
        # Update parameters
        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        # dA_prev = ∂L / ∂A_prev
        #         = (∂L / ∂Z) * (∂Z / ∂A_prev)
        #         = dZ * W
        return np.dot(dZ, self.W.T)


class Activation(Layer):
    def __init__(self, name, function):
        self.name = name
        self.function = function

    def name(self):
        return self.name

    def num_parameters(self):
        return 0

    def shape(self):
        return None, None

    def forward(self, Z, training=True):
        """Forward pass
        """
        if training:
            self.Z = Z
        return self.function(Z)

    def backward(self, dA):
        """Backward pass
        """
        # dZ = ∂L / ∂Z
        #    = (∂L / ∂A) * (∂A / ∂Z)
        #    = dA * g'(Z)
        return np.multiply(dA, self.function.gradient(self.Z))

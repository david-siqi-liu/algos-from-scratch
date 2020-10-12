import matplotlib.pyplot as plt
from terminaltables import AsciiTable

from algosfromscratch.deep_learning.activation_functions import *
from algosfromscratch.deep_learning.layers import *
from algosfromscratch.deep_learning.loss_functions import *


class MLPClassifier:

    def __init__(self, hidden_layer_sizes=(100,), hidden_activation='relu', out_activation='softmax',
                 learning_rate=0.001, max_iter=200, loss_function='cross-entropy', random_state=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_activation = self._get_activation_function(hidden_activation)
        self.out_activation = self._get_activation_function(out_activation)
        self.loss = self._get_loss_function(loss_function)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state

    def _get_activation_function(self, activation_function):
        if activation_function == 'identity':
            return Identity()
        elif activation_function == 'sigmoid':
            return Sigmoid()
        elif activation_function == 'softmax':
            return Softmax()
        elif activation_function == 'tanh':
            return TanH()
        elif activation_function == 'relu':
            return ReLU()
        else:
            raise NotImplementedError

    def _get_loss_function(self, loss_function):
        if loss_function == 'quadratic':
            return QuadraticLoss()
        elif loss_function == 'cross-entropy':
            return CrossEntropyLoss()
        else:
            raise NotImplementedError

    def _initialize_weights(self, factor, n_in, n_out):
        init_bound = np.sqrt(factor / (n_in + n_out))
        W = self.random_instance.uniform(-init_bound, init_bound, (n_in, n_out))
        b = self.random_instance.uniform(-init_bound, init_bound, (1, n_out))
        return W, b

    def _initialize(self, X):
        # Re-create random instance
        self.random_instance = np.random.mtrand._rand if not self.random_state else np.random.RandomState(
            self.random_state)
        # Number of input features
        self.n_features = X.shape[1]
        # Layers
        self.layers = []
        # Input layer and activation
        init_weights = self._initialize_weights(4, self.n_features, self.hidden_layer_sizes[0])
        self.layers.append(LinearLayer(name='input_layer', weights=init_weights))
        self.layers.append(Activation(name='input_activation', function=self.hidden_activation))
        # Hidden layers and activations
        for i in range(len(self.hidden_layer_sizes) - 1):
            init_weights = self._initialize_weights(4, self.hidden_layer_sizes[i], self.hidden_layer_sizes[i + 1])
            self.layers.append(LinearLayer(name='hidden_layer_{}'.format(i + 1), weights=init_weights))
            self.layers.append(Activation(name='hidden_activation_{}'.format(i + 1), function=self.hidden_activation))
        # Output layer and activation
        init_weights = self._initialize_weights(4, self.hidden_layer_sizes[-1], 1)
        self.layers.append(LinearLayer(name='output_layer', weights=init_weights))
        self.layers.append(Activation(name='output_activation', function=self.out_activation))
        assert len(self.layers) == (len(self.hidden_layer_sizes) + 1) * 2
        # Print summary
        self.summary()

    def _forward(self, X, training=True):
        output = X
        for layer in self.layers:
            output = layer.forward(output, training)
        return output

    def _backward(self, grad):
        for layer in reversed(self.layers):
            if layer.type() == 'LinearLayer':
                grad = layer.backward(grad, self.learning_rate)
            elif layer.type() == 'Activation':
                grad = layer.backward(grad)
            else:
                raise NotImplementedError

    def fit(self, X, y, verbose=False):
        # Reshape y if necessary
        if len(y.shape) == 1:
            y = np.reshape(y, (-1, 1))
        # Initialize layers with random weights and biases
        self._initialize(X)
        # Run and keep track of average costs
        self.training_costs = []
        for i in range(self.max_iter):
            # Forward pass
            y_pred = self._forward(X, training=True)
            # Compute average loss
            cost = np.mean(self.loss(y, y_pred))
            assert cost.shape == ()
            if verbose:
                print("Iteration {:d}, loss = {:.8f}".format(i + 1, cost))
            # Early stopping
            if self.training_costs and abs(cost) > abs(self.training_costs[-1]):
                break
            # Add to list
            self.training_costs.append(cost)
            # Compute gradient of the cost w.r.t. y_pred
            grad = self.loss.gradient(y, y_pred)
            # Backward pass
            self._backward(grad)

    def predict_proba(self, X):
        return np.reshape(self._forward(X, training=False), (-1,))

    def predict(self, X):
        return np.round(self.predict_proba(X)).astype(int)

    def summary(self, name='Model Summary'):
        print(AsciiTable([[name]]).table)
        # Iterate through each layer
        table_data = [["Layer Name", "Layer Type", "Shape - Weights", "Shape - Biases", "Parameters"]]
        total_parameters = 0
        for layer in self.layers:
            name = layer.name
            type = layer.type()
            num_parameters = layer.num_parameters()
            W_shape, b_shape = layer.shape()
            table_data.append([name, type, str(W_shape), str(b_shape), str(num_parameters)])
            total_parameters += num_parameters
        # Print
        print(AsciiTable(table_data).table)
        print("Total Parameters: {:,d}\n".format(total_parameters))

    def plot_training_costs(self):
        n = len(self.training_costs)
        _ = plt.plot(range(n), self.training_costs, label="Training Costs")
        plt.title("Training Cost Plot")
        plt.ylabel('Costs')
        plt.xlabel('Iterations')
        plt.show()

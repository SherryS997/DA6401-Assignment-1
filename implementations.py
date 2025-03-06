import numpy as np
import time

class ActivationFunctions:
    """Provides static methods for common activation functions and their derivatives."""
    @staticmethod
    def identity(x, derivative = False):
        """Identity activation function."""
        if derivative:
            return np.ones_like(x)
        return x

    @staticmethod
    def sigmoid(x, derivative = False):
        """Sigmoid activation function."""
        if derivative:
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x, derivative = False):
        """Tanh activation function."""
        if derivative:
            return 1 - np.tanh(x) ** 2
        return np.tanh(x)

    @staticmethod
    def relu(x, derivative = False):
        """ReLU activation function."""
        if derivative:
            return np.where(x > 0, 1, 0)
        return np.maximum(0, x)


class LossFunctions:
    """Provides static methods for common loss functions and their derivatives."""
    @staticmethod
    def mean_squared_error(y_true, y_pred, derivative = False):
        """Mean Squared Error loss function."""
        if derivative:
            return 2 * (y_pred - y_true) / y_true.shape[0]
        return np.mean(np.sum((y_true - y_pred) ** 2, axis=1))

    @staticmethod
    def cross_entropy(y_true, y_pred, derivative = False, epsilon = 1e-10):
        """Cross Entropy loss function."""
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        if derivative:
            return -y_true / y_pred + (1 - y_true) / (1 - y_pred)

        return -np.mean(np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), axis=1))


class Optimizers:
    """Provides static methods for various optimization algorithms."""
    @staticmethod
    def sgd(params, grads, config):
        """Stochastic Gradient Descent optimizer."""
        learning_rate = config.get('learning_rate', 0.01)
        weight_decay = config.get('weight_decay', 0.0)

        for i, param in enumerate(params):
            if weight_decay > 0:
                grads[i] += weight_decay * param

            params[i] -= learning_rate * grads[i]

        return params, config

    @staticmethod
    def momentum(params, grads, config):
        """Momentum optimizer."""
        learning_rate = config.get('learning_rate', 0.01)
        momentum = config.get('momentum', 0.9)
        weight_decay = config.get('weight_decay', 0.0)

        if 'velocity' not in config:
            config['velocity'] = [np.zeros_like(param) for param in params]

        for i, param in enumerate(params):
            if weight_decay > 0:
                grads[i] += weight_decay * param

            config['velocity'][i] = momentum * config['velocity'][i] - learning_rate * grads[i]

            params[i] += config['velocity'][i]

        return params, config

    @staticmethod
    def nag(params, grads, config):
        """Nesterov Accelerated Gradient optimizer."""
        learning_rate = config.get('learning_rate', 0.01)
        momentum = config.get('momentum', 0.9)
        weight_decay = config.get('weight_decay', 0.0)

        if 'velocity' not in config:
            config['velocity'] = [np.zeros_like(param) for param in params]

        for i, param in enumerate(params):
            if weight_decay > 0:
                grads[i] += weight_decay * param

            v_prev = config['velocity'][i].copy()
            config['velocity'][i] = momentum * config['velocity'][i] - learning_rate * grads[i]

            params[i] += -momentum * v_prev + (1 + momentum) * config['velocity'][i]

        return params, config

    @staticmethod
    def rmsprop(params, grads, config):
        """RMSprop optimizer."""
        learning_rate = config.get('learning_rate', 0.01)
        beta = config.get('beta', 0.9)
        epsilon = config.get('epsilon', 1e-8)
        weight_decay = config.get('weight_decay', 0.0)

        if 'square_grad' not in config:
            config['square_grad'] = [np.zeros_like(param) for param in params]

        for i, param in enumerate(params):
            if weight_decay > 0:
                grads[i] += weight_decay * param

            config['square_grad'][i] = beta * config['square_grad'][i] + (1 - beta) * grads[i]**2

            params[i] -= learning_rate * grads[i] / (np.sqrt(config['square_grad'][i]) + epsilon)

        return params, config

    @staticmethod
    def adam(params, grads, config):
        """Adam optimizer."""
        learning_rate = config.get('learning_rate', 0.01)
        beta1 = config.get('beta1', 0.9)
        beta2 = config.get('beta2', 0.999)
        epsilon = config.get('epsilon', 1e-8)
        weight_decay = config.get('weight_decay', 0.0)

        if 't' not in config:
            config['t'] = 0
            config['m'] = [np.zeros_like(param) for param in params]
            config['v'] = [np.zeros_like(param) for param in params]

        config['t'] += 1

        for i, param in enumerate(params):
            if weight_decay > 0:
                grads[i] += weight_decay * param

            config['m'][i] = beta1 * config['m'][i] + (1 - beta1) * grads[i]

            config['v'][i] = beta2 * config['v'][i] + (1 - beta2) * grads[i]**2

            m_corrected = config['m'][i] / (1 - beta1**config['t'])
            v_corrected = config['v'][i] / (1 - beta2**config['t'])

            params[i] -= learning_rate * m_corrected / (np.sqrt(v_corrected) + epsilon)

        return params, config

    @staticmethod
    def nadam(params, grads, config):
        """Nadam optimizer (Adam with Nesterov momentum)."""
        learning_rate = config.get('learning_rate', 0.01)
        beta1 = config.get('beta1', 0.9)
        beta2 = config.get('beta2', 0.999)
        epsilon = config.get('epsilon', 1e-8)
        weight_decay = config.get('weight_decay', 0.0)

        if 't' not in config:
            config['t'] = 0
            config['m'] = [np.zeros_like(param) for param in params]
            config['v'] = [np.zeros_like(param) for param in params]

        config['t'] += 1

        for i, param in enumerate(params):
            if weight_decay > 0:
                grads[i] += weight_decay * param

            config['m'][i] = beta1 * config['m'][i] + (1 - beta1) * grads[i]

            config['v'][i] = beta2 * config['v'][i] + (1 - beta2) * grads[i]**2

            m_corrected = config['m'][i] / (1 - beta1**config['t'])
            v_corrected = config['v'][i] / (1 - beta2**config['t'])

            m_update = beta1 * m_corrected + (1 - beta1) * grads[i] / (1 - beta1**config['t'])

            params[i] -= learning_rate * m_update / (np.sqrt(v_corrected) + epsilon)

        return params, config


class FeedForwardNeuralNetwork:
    """A simple feedforward neural network class."""
    def __init__(self, input_size, output_size,
                 hidden_layers = [128],
                 activation = 'sigmoid',
                 loss = 'cross_entropy',
                 optimizer = 'sgd',
                 learning_rate = 0.01,
                 weight_init = 'random',
                 **kwargs):
        """Initializes the FeedForwardNeuralNetwork."""
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers

        self.activation_name = activation.lower()
        if self.activation_name == 'identity':
            self.activation_fn = ActivationFunctions.identity
        elif self.activation_name == 'sigmoid':
            self.activation_fn = ActivationFunctions.sigmoid
        elif self.activation_name == 'tanh':
            self.activation_fn = ActivationFunctions.tanh
        elif self.activation_name == 'relu':
            self.activation_fn = ActivationFunctions.relu
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        self.loss_name = loss.lower()
        if self.loss_name == 'mean_squared_error':
            self.loss_fn = LossFunctions.mean_squared_error
        elif self.loss_name == 'cross_entropy':
            self.loss_fn = LossFunctions.cross_entropy
        else:
            raise ValueError(f"Unsupported loss function: {loss}")

        self.optimizer_name = optimizer.lower()
        if self.optimizer_name == 'sgd':
            self.optimizer = Optimizers.sgd
        elif self.optimizer_name == 'momentum':
            self.optimizer = Optimizers.momentum
        elif self.optimizer_name == 'nag':
            self.optimizer = Optimizers.nag
        elif self.optimizer_name == 'rmsprop':
            self.optimizer = Optimizers.rmsprop
        elif self.optimizer_name == 'adam':
            self.optimizer = Optimizers.adam
        elif self.optimizer_name == 'nadam':
            self.optimizer = Optimizers.nadam
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        self.optimizer_config = {
            'learning_rate': learning_rate,
            'momentum': kwargs.get('momentum', 0.9),
            'beta': kwargs.get('beta', 0.9),
            'beta1': kwargs.get('beta1', 0.9),
            'beta2': kwargs.get('beta2', 0.999),
            'epsilon': kwargs.get('epsilon', 1e-8),
            'weight_decay': kwargs.get('weight_decay', 0.0)
        }

        self.layers_dims = [input_size] + hidden_layers + [output_size]

        self.params = {}
        self.initialize_weights(weight_init)

        self.train_loss_history = []
        self.val_loss_history = []
        self.train_accuracy_history = []
        self.val_accuracy_history = []

    def initialize_weights(self, weight_init):
        """Initializes weights and biases based on the specified method."""
        np.random.seed(42)

        for l in range(1, len(self.layers_dims)):
            if weight_init.lower() == 'xavier':
                scale = np.sqrt(2.0 / (self.layers_dims[l-1] + self.layers_dims[l]))
                self.params[f'W{l}'] = np.random.randn(self.layers_dims[l-1], self.layers_dims[l]) * scale
            else:
                self.params[f'W{l}'] = np.random.randn(self.layers_dims[l-1], self.layers_dims[l]) * 0.01

            self.params[f'b{l}'] = np.zeros((1, self.layers_dims[l]))

    def forward_propagation(self, X):
        """Performs forward propagation through the network."""
        cache = {}
        A = {'A0': X}

        for l in range(1, len(self.layers_dims) - 1):
            Z = np.dot(A[f'A{l-1}'], self.params[f'W{l}']) + self.params[f'b{l}']
            cache[f'Z{l}'] = Z

            A[f'A{l}'] = self.activation_fn(Z)

        l = len(self.layers_dims) - 1
        Z = np.dot(A[f'A{l-1}'], self.params[f'W{l}']) + self.params[f'b{l}']
        cache[f'Z{l}'] = Z

        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        A[f'A{l}'] = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

        return cache, A

    def backward_propagation(self, y, cache, A):
        """Performs backward propagation to calculate gradients."""
        m = y.shape[0]
        L = len(self.layers_dims) - 1
        gradients = {}

        dZ = A[f'A{L}'] - y

        for l in range(L, 0, -1):
            gradients[f'dW{l}'] = (1/m) * np.dot(A[f'A{l-1}'].T, dZ)
            gradients[f'db{l}'] = (1/m) * np.sum(dZ, axis=0, keepdims=True)

            if l > 1:
                dA_prev = np.dot(dZ, self.params[f'W{l}'].T)
                dZ = dA_prev * self.activation_fn(cache[f'Z{l-1}'], derivative=True)

        return gradients

    def update_parameters(self, gradients):
        """Updates network parameters using the chosen optimizer."""
        params_list = []
        grads_list = []

        for l in range(1, len(self.layers_dims)):
            params_list.append(self.params[f'W{l}'])
            params_list.append(self.params[f'b{l}'])
            grads_list.append(gradients[f'dW{l}'])
            grads_list.append(gradients[f'db{l}'])

        updated_params, self.optimizer_config = self.optimizer(params_list, grads_list, self.optimizer_config)

        idx = 0
        for l in range(1, len(self.layers_dims)):
            self.params[f'W{l}'] = updated_params[idx]
            idx += 1
            self.params[f'b{l}'] = updated_params[idx]
            idx += 1

    def fit(self, X_train, y_train,
            epochs = 10, batch_size = 32,
            X_val = None, y_val = None):
        """Trains the neural network on the given training data."""
        m = X_train.shape[0]

        for epoch in range(epochs):
            start_time = time.time()
            epoch_loss = 0

            permutation = np.random.permutation(m)
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]

            for i in range(0, m, batch_size):
                end = min(i + batch_size, m)
                X_batch = X_shuffled[i:end]
                y_batch = y_shuffled[i:end]

                cache, A = self.forward_propagation(X_batch)

                batch_loss = self.loss_fn(y_batch, A[f'A{len(self.layers_dims) - 1}'])
                epoch_loss += batch_loss * (end - i) / m

                gradients = self.backward_propagation(y_batch, cache, A)

                self.update_parameters(gradients)

            self.train_loss_history.append(epoch_loss)

            y_train_pred = self.predict(X_train)
            train_accuracy = np.mean(np.argmax(y_train, axis=1) == np.argmax(y_train_pred, axis=1))
            self.train_accuracy_history.append(train_accuracy)

            val_loss = None
            val_accuracy = None
            if X_val is not None and y_val is not None:
                val_cache, val_A = self.forward_propagation(X_val)
                val_loss = self.loss_fn(y_val, val_A[f'A{len(self.layers_dims) - 1}'])
                self.val_loss_history.append(val_loss)

                y_val_pred = self.predict(X_val)
                val_accuracy = np.mean(np.argmax(y_val, axis=1) == np.argmax(y_val_pred, axis=1))
                self.val_accuracy_history.append(val_accuracy)

            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - loss: {epoch_loss:.4f} - accuracy: {train_accuracy:.4f}",
                  end="")
            if val_loss is not None:
                print(f" - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}", end="")
            print()

        return {
            'train_loss': self.train_loss_history,
            'val_loss': self.val_loss_history,
            'train_accuracy': self.train_accuracy_history,
            'val_accuracy': self.val_accuracy_history
        }

    def predict(self, X):
        """Predicts output for the given input data."""
        _, A = self.forward_propagation(X)
        return A[f'A{len(self.layers_dims) - 1}']

    def evaluate(self, X, y):
        """Evaluates the model on the given data and labels."""
        y_pred = self.predict(X)

        loss = self.loss_fn(y, y_pred)

        accuracy = np.mean(np.argmax(y, axis=1) == np.argmax(y_pred, axis=1))

        return loss, accuracy


def preprocess_data(X_train, y_train, X_test, y_test):
    """Preprocesses the input data for training and testing."""
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    num_classes = 10
    y_train_onehot = np.zeros((y_train.size, num_classes))
    y_train_onehot[np.arange(y_train.size), y_train] = 1

    y_test_onehot = np.zeros((y_test.size, num_classes))
    y_test_onehot[np.arange(y_test.size), y_test] = 1

    return X_train, y_train_onehot, X_test, y_test_onehot
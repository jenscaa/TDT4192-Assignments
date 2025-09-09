import numpy as np


class LogisticRegression:

    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.losses = []
        self.train_accuracies = []

    def reset(self, learning_rate=0.1, epochs=1000):
        """
        Resets the model's parameters to re-train from scratch.

        :param learning_rate: The step size for gradient descent
        :param epochs: Number of iterations to optimize the weights
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.losses = []
        self.train_accuracies = []

    def _sigmoid(self, x):
        """
        Applies the sigmoid activation function.

        :param x: Input
        :return: Sigmoid-transformed values between 0 and 1
        """
        return 1.0 / (1.0 + np.exp(-x))

    def _compute_loss(self, y, y_pred):
        """
        Computes the binary cross-entropy loss.

        :param y: True labels (array<m>)
        :param y_pred: Predicted probabilities (array<m>)
        :return: Mean binary cross-entropy loss
        """
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def _compute_gradients(self, x, y, y_pred):
        """
        Computes gradients of the weights and bias using binary cross-entropy loss.

        :param x: Feature matrix (array<m,n>)
        :param y: True labels (array<m>)
        :param y_pred: Predicted probabilities (array<m>)
        :return: grad_w (array<n>), grad_b (scalar)
        """

        # BCE: Binary Cross-Entropy
        # L = -(1 / N) * Σ [ y * log(y_pred) + (1 - y) * log(1 - y_pred) ]
        # where: y_pred = σ(z), z = Xw + b, σ(z) = 1 / (1 + e ^ {-z})
        # dLdw = (1 / N) * X * (y_pred - y)
        # dLdb = (1 / N) * Σ (y_pred - y)

        N = x.shape[0]
        residuals = y - y_pred
        grad_w = -(x.T @ residuals) / N
        grad_b = -np.sum(residuals) / N
        return grad_w, grad_b

    def _update_parameters(self, grad_w, grad_b):
        """
        Updates weights and bias using gradient descent.

        :param grad_w: Gradient for weights (array<n>)
        :param grad_b: Gradient for bias (scalar)
        """
        self.weights -= self.learning_rate * grad_w
        self.bias -= self.learning_rate * grad_b

    def accuracy(self, true_values, predictions):
        """
        Calculates classification accuracy.

        :param true_values: True labels (array<m>)
        :param predictions: Predicted labels (array<m>)
        :return: Accuracy score (float between 0 and 1)
        """
        return np.mean(true_values == predictions)

    def fit(self, x, y):
        """
        Estimates parameters for the classifier

        :param x: Feature matrix (array<m,n>)
        :param y: True labels (array<m>)
        """
        self.weights = np.zeros(x.shape[1])
        self.bias = 0

        # Gradient Descent
        for _ in range(self.epochs):
            lin_model = np.matmul(self.weights, x.transpose()) + self.bias
            y_pred = self._sigmoid(lin_model)
            grad_w, grad_b = self._compute_gradients(x, y, y_pred)
            self._update_parameters(grad_w, grad_b)

            loss = self._compute_loss(y, y_pred)
            pred_to_class = [1 if _y > 0.5 else 0 for _y in y_pred]
            self.train_accuracies.append(self.accuracy(y, pred_to_class))
            self.losses.append(loss)

    def predict(self, x):
        """
        Generates predictions for given input features.

        :param x: Feature matrix (array<m,n>)
        :return: Predicted class labels (array<m>)
        """
        lin_model = np.matmul(x, self.weights) + self.bias
        y_pred = self._sigmoid(lin_model)
        return [1 if _y > 0.5 else 0 for _y in y_pred]

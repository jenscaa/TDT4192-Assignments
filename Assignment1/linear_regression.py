import numpy as np


class LinearRegression:

    def __init__(self, learning_rate=0.0001, epochs=500):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.losses = []
        self.train_accuracies = []

    def _compute_gradients(self, X, y):
        """
        Compute gradients by using Mean Squared Error (MSE) as the loss function.

        :param X: Feature matrix (array<m,n>) of floats with
                m rows (#samples) and n columns (#features)
        :param y: A vector (array<m>) of floats

        :return: Gradient to the Weights, gradient to the bias
        """

        N = X.shape[0]
        y_pred = X @ self.weights + self.bias
        residual = y - y_pred

        # MSE: Mean Square error
        # L = 1 / N * (y - y_pred) ** 2
        # dLdw = 2 / N * (y - (wX + b))) * (-X) = - (2 / N) * (X * (y - (wX + b)))
        # dLdb = 2 / N * (y - (wX + b))) * (-1) = - (2 / N) * (y - (wX + b))

        grad_w = -(2 / N) * (X.T @ residual)
        grad_b = -(2 / N) * np.sum(residual)
        return grad_w, grad_b

    def _update_parameters(self, grad_w, grad_b):
        """
        Updates weights and bias for the model

        :param grad_w: Gradient matrix for the weights
        :param grad_b: Gradient scalar for the bias
        """

        self.weights -= self.learning_rate * grad_w
        self.bias -= self.learning_rate * grad_b

    def fit(self, X, y):
        """
        Estimates parameters for the classifier

        :param X: Feature matrix (array<m,n>) of floats with
                m rows (#samples) and n columns (#features)
        :param y: Vector (array<m>) of floats
        """

        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        # Gradient Decent
        for _ in range(self.epochs):
            grad_w, grad_b = self._compute_gradients(X, y)
            self._update_parameters(grad_w, grad_b)

    def predict(self, X):
        """
        Generates predictions

        Note: should be called after .fit()

        :param X: Feature matrix of floats with
                m rows (#samples) and n columns (#features)
        :return: Array of length m floats
        """

        lin_model = np.matmul(X, self.weights) + self.bias
        return lin_model

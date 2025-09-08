import numpy as np


class LinearRegression():

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

        :param X:
        :param y:
        :return: gradient to the Weights, gradient to the bias
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
        self.weights -= self.learning_rate * grad_w
        self.bias -= self.learning_rate * grad_b

    def fit(self, X, y):
        """
        Estimates parameters for the classifier

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
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

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)

        Returns:
            A length m array of floats
        """

        lin_model = np.matmul(X, self.weights) + self.bias
        return lin_model

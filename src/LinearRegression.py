"""Implementing Linear Regression algorithm from scratch"""
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple, Union

from sklearn.model_selection import train_test_split
from utils import BaseModel, mean_squared_error_fn
from sklearn.datasets import load_boston, load_diabetes



Matrix = Union[List, Tuple, np.ndarray]
# Loading Data

# Linear Regression Implementation
class LinearRegression(BaseModel):
    """Linear Regression implementation 
        y = ax + b
    """
    def __init__(self, learning_rate: float = 0.05, 
                        iterations: int = 1000, 
                        name: str = "Linear Regression Model") -> None:
        super().__init__(name)
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X: Matrix, y: Matrix) -> None:
        """Fitting linear regression model with data

        Args:
            X (Matrix): Training data
            y (Matrix): Trainign label

        Returns:
            _type_: _description_
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float64)
        if not isinstance(y, np.ndarray):
            y = np.array(y, dtype=np.float64)
        self.m, self.n = X.shape
        self.W: Matrix = np.zeros(self.n)
        self.b: float = 0
        self.X: Matrix = X
        self.y: Matrix = y
        self.losses = []
        for iteration in range(self.iterations):
            self.update_weights()
        return self

    def update_weights(self) -> None:
        """Update internal weights based on model's algorithm
        """
        y_pred = self.predict(self.X)
        loss = mean_squared_error_fn(y_pred, self.y)
        self.losses.append(loss)
        dW = -(2 * (self.X.T).dot(y_pred- self.y)) / self.m
        db = - 2 * np.sum(y_pred - self.y) / self.m
        # update weights
        self.W = self.W + self.learning_rate * dW
        self.b = self.b + self.learning_rate * db
        return self
    
    def predict(self, X: Matrix) -> Matrix:
        """Predicting unseen data using trained weights

        Args:
            X (Matrix): Unseen data

        Returns:
            Matrix: predictions
        """
        return X.dot(self.W) + self.b

    def visualize_losses(self) -> None:
        plt.plot(range(1, self.iterations + 1), self.losses)
        plt.title("MSE per iteration")
        plt.xlabel("Number of Iterations")
        plt.ylabel("MSE")
        plt.show()


# Training on data
def main():
    """Train LR from scratch on iris dataset
    """
    data = load_diabetes()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr.visualize_losses()

if __name__ == "__main__":
    main()
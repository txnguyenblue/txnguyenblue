"""Implementing Linear Regression algorithm from scratch"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging

from typing import List, Tuple, Union
from config import CONFIG

sys.path.insert(1, str(CONFIG.utils))


from sklearn.model_selection import train_test_split
from utilities import BaseModel, mean_squared_error_fn, visualize_losses, Matrix, Vector
from sklearn.datasets import load_boston, load_diabetes
from logger import LOGGER



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

    def plot_loss(self) -> None:
        visualize_losses(self.iterations, self.losses)



# Training on data
def main():
    """Train LR from scratch on iris dataset
    """
    LOGGER.info("Loading diabetes data from sklearn...")
    data = load_diabetes()
    X, y = data.data, data.target
    LOGGER.info("Split dataset into train, test sets with ratio 0.8")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    LOGGER.info("Train a linear regression model to predict diabetes")
    lr = LinearRegression()
    LOGGER.info("Done training!")
    LOGGER.info("Visualizing losses per epoch")
    lr.fit(X_train, y_train)
    lr.plot_loss()

if __name__ == "__main__":
    main()
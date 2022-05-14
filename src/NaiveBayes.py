"""Naive Bayes implementation from scratch
@Author: https://towardsdatascience.com/implementing-naive-bayes-from-scratch-df5572e042ac"""

from operator import ge
import sys
import numpy as np
import pandas as pd

import time
from sklearn.model_selection import train_test_split

from config import CONFIG
from sklearn.datasets import load_iris

sys.path.insert(1, str(CONFIG.src))
sys.path.insert(2, str(CONFIG.utils))

from utilities import \
(get_accuracy, visualize_losses, BaseModel, Matrix, Vector, get_numpy_instance)

from logger import LOGGER


class NaiveBayes(BaseModel):
    """Implementing Naive Bayes classifier
    P(Class|data) = P(data|Class)P(Class) / P(data)

    Args:
        BaseModel (_type_): _description_
    """
    def __init__(self, name: str = "Naive Bayes Model") -> None:
        super().__init__(name)
        
    #TODO: Write test for fit function
    def fit(self, X: Matrix, y: Vector) -> None:
        cond1a = isinstance(X, pd.DataFrame)
        cond1b = isinstance(y, pd.DataFrame)
        cond2a = isinstance(X, pd.Series)
        cond2b = isinstance(y, pd.Series)
        cond1 = (cond1a | cond1b)
        cond2 = (cond2a | cond2b)
        
        if cond1 or cond2:
            X = X.values
            y = y.values
        else:
            X = get_numpy_instance(X) 
            y = get_numpy_instance(y)
        self.n_samples, self.n_features = len(X), len(X[0])
        self.n_classes = len(np.unique(y))

        self.mean = np.zeros((self.n_classes, self.n_features))
        self.variance = np.zeros((self.n_classes, self.n_features))
        self.priors = np.zeros(self.n_classes)

        for c in range(self.n_classes):
            X_c = X[y == c]
            self.mean[c, :] = np.mean(X_c, axis=0)
            self.variance[c, :] = np.var(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / self.n_samples
        



    def predict(self, X) -> None:
        y_hat = [self.get_class_probability(x) for x in X]
        return np.array(y_hat)
    
    def get_class_probability(self, x: Matrix) -> float:
        """output probabilities vector of a prediction

        Args:
            X (Matrix): _description_

        Returns:
            float: _description_
        """
        posteriors = []
        for c in range(self.n_classes):
            mean = self.mean[c]
            variance = self.variance[c]
            prior = np.log(self.priors[c])

            posterior = np.sum(np.log(self.gaussian_density(x, mean, variance)))
            posterior = prior + posterior
            posteriors.append(posterior)
        return np.argmax(posteriors)

    def gaussian_density(self, x: float, mean: float, var: float) -> float:
        """_summary_

        Args:
            X (Matrix): _description_
            mean (float): _description_
            var (float): _description_

        Returns:
            Matrix: _description_
        """
        const = 1 / np.sqrt(var * 2 * np.pi)
        proba = np.exp(-0.5 * ((x - mean) ** 2 / var))
        return const * proba




#TODO: implement argparse logic to save reports into report folder
def main():
    LOGGER.info("Starting to train a naive bayes model on the iris dataset...")
    time.sleep(5)
    LOGGER.info("Load iris dataset")
    iris = load_iris()
    X = iris.data
    y = iris.target

    LOGGER.info("Slitting data into train and test set with ratio 0.8...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)
    accuracy = get_accuracy(predictions, y_test)
    LOGGER.info(f"Naive bayes accuracy: {accuracy}")

if __name__ == "__main__":
    main()
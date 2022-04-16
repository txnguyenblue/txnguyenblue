"""Utility functions """
import abc
import numpy as np
import pandas as pd
from typing import Tuple, List, Union, Any
from matplotlib import pyplot as plt

Matrix = Union[Tuple[Tuple[Any]], List[List[Any]], np.ndarray]
Vector = Union[Tuple[Any], List[Any], np.ndarray]

class BaseModel(metaclass=abc.ABCMeta):
    
    def __init__(self, name: str = "Base Model") -> None:
        self.name = name

    @abc.abstractmethod
    def fit(self) -> None:
        pass

    @abc.abstractmethod
    def predict(self) -> None:
        pass

    # @abc.abstractclassmethod
    # def update_weights(self) -> None:
    #     pass

    def __str__(self) -> None:
        return f"{self.name} for machine learning"
    
class Eval(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        pass
    @abc.abstractclassmethod
    def plot(self, predictions: Vector, ground_truths: Vector) -> None:
        pass

    def get_confusion_matrix(self, predictions: Vector, ground_truths: Vector) -> pd.DataFrame:
        """computing confusion matrix

        Args:
            predictions (Vector): predictions from the model
            ground_truths (Vector): the labels
        Return:
            (pd.DataFrame): confusion matrix in dataframe
        """
        df_confusion = pd.crosstab(ground_truths, predictions)

class Visualization(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        pass
class TestModel(BaseModel):

    def __init__(self, name: str = "Test Model") -> None:
        super().__init__(name)

    def fit(self):
        pass

    def predict(self) -> None:
        return super().predict()

class Loss(metaclass=abc.ABCMeta):

    def __init__(self) -> None:
        pass

    @abc.abstractclassmethod
    def __loss_fn(self) -> float:
        pass
#==========

def mean_squared_error_fn(y_pred: Vector, y_true: Vector):
    """Calculting linear regression loss

    Args:
        y_pred (Vector): predictions
        y (Vector): ground_truth

    Returns:
        float: least squared error
    
    Examples
    --------
    >>> y_true = [0, 1, 2, 3, 4]
    >>> y_pred = [0, 1, 2, 3, 4]
    >>> print(mean_squared_error_fn(y_pred, y_true))

    0.0

    >>> y_test = [0, 2, 3, 1]
    >>> y_pred = [0, 3, 4, 1]
    >>> print(mean_squared_error_fn(y_pred, y_test))

    0.5
    """
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    m = len(y_pred)
    # loss = 1 / (m) * np.sum(np.power((y_pred - y_true), 2))
    loss = (np.square(y_true - y_pred)).mean()
    return loss

def gradient_descent_lr(X: Matrix, y_true: Vector, W):
    y_pred = predict_lr(X, W)
    pass

def visualize_losses(epochs: int, losses: Vector) -> None:
    """Visualize model's losses over each epoch

    Args:
        epochs (int): the number of training epochs
        losses (Vector): loss values per training epoch
    """
    plt.plot(range(1, epochs + 1), losses)
    plt.title("MSE per iteration")
    plt.xlabel("Number of Iterations")
    plt.ylabel("MSE")
    plt.show()

def get_numpy_instance(X: Matrix):
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    return X

def get_accuracy(y_true: Vector, y_hat: Vector) -> float:
    """get accuracy from predictions and ground truths

    Args:
        y_true (Vector): ground truths vector
        y_hat (Vector): predictions vector

    Returns:
        float: accuracy score
    """

    return np.sum(y_true == y_hat) / len(y_true)

#===============
class GradientDescent(metaclass=abc.ABCMeta):
    pass



if __name__ == "__main__":
    test_abstract = TestModel()
    print(test_abstract)
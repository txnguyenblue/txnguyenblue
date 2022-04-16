"""Utility functions """
import abc
import numpy as np
from typing import Tuple, List, Union, Any

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

    @abc.abstractclassmethod
    def update_weights(self) -> None:
        pass

    def __str__(self) -> None:
        return f"{self.name} for machine learning"
    

class TestModel(BaseModel):

    def __init__(self, name: str = "Test Model") -> None:
        super().__init__(name)

    def fit(self):
        pass

    def predict(self) -> None:
        return super().predict()
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
    >>> 
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

def visualize_losses(self) -> None:
        plt.plot(range(1, self.iterations + 1), self.losses)
        plt.title("MSE per iteration")
        plt.xlabel("Number of Iterations")
        plt.ylabel("MSE")
        plt.show()

#===============
class GradientDescent(metaclass=abc.ABCMeta):
    pass



if __name__ == "__main__":
    test_abstract = TestModel()
    print(test_abstract)
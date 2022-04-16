"""Utility functions """
import abc
import numpy as np
import pandas as pd
import matplotlib
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
    
class Eval():
    """evaluation tools for machine learning
    """
    def __init__(self) -> None:
        pass

    def get_confusion_matrix(self, predictions: Vector, ground_truths: Vector) -> pd.DataFrame:
        """computing confusion matrix

        Args:
            predictions (Vector): predictions from the model
            ground_truths (Vector): the labels
        Return:
            (pd.DataFrame): confusion matrix in dataframe

        Examples
        >>> preds = [1,2,3,1,2,1,1,1]
        >>> labels = [1,2,3,2,3,1,1,2]
        >>> print(get_confusion_matrix(predictions, ground_truths))

        Predicted  1  2  3  
        Actual
        1          3  2  0        
        2          0  1  1        
        3          0  0  1              
        """
        df_confusion = pd.crosstab(ground_truths, predictions)
        return df_confusion

    
class Visualization():
    """visualization tools for machine learning
    """
    def __init__(self, fig_size: Tuple[int] = (10, 5)) -> None:
        self.fig_size = fig_size
        eval = Eval()
    
    def plot_confusion_matrix(self, predictions: Vector, ground_truths: Vector) -> matplotlib.axes._subplots.AxesSubplot:
        """plot confusion matrix

        Args:
            predictions (Vector): _description_
            ground_truths (Vector): _description_

        Returns:
            matplotlib.axes._subplots.AxesSubplot: _description_
        
        Examples
        --------

        >>> predictions = [1,2,3,1,2,2,2] 
        >>> ground_truths = [1,2,1,1,2,2,2]
        >>> plot_confusion_matrix(predictions, ground_truths)

        matplotlib.axes._subplots.AxesSubplot
        """
        f, ax = plt.subplots(figsize=self.fig_size)
        df_confusion = eval.get_confusion_matrix(predictions, ground_truths)
        ax.matshow(df_confusion)
        for (i, j), value in np.ndenumerate(df_confusion.values):
            ax.text(j, i, f"{value}", ha="center", va="center")
        plt.show()
        return ax
        
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
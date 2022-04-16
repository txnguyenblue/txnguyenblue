import sys
import numpy as np
import pytest

from sklearn.metrics import mean_squared_error
from config import CONFIG

sys.path.insert(1, str(CONFIG.src))

from utils import mean_squared_error_fn


def generate_random_data(start: int = 0, stop: int = 10, step: float = 0.6):
    y_true = np.arange(start, stop, step)
    y_pred = np.arange(start, stop, step)
    return y_true, y_pred

data = []
for i in range(10):
    y_true, y_pred = generate_random_data()
    data.append((y_true, y_pred))


@pytest.mark.parametrize("y_true, y_pred", data)
def test_mean_squared_error(y_true, y_pred):
    assert mean_squared_error(y_true, y_pred) == mean_squared_error_fn(y_true, y_pred)





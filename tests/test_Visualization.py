import sys
import numpy as np
import pytest

from sklearn.metrics import mean_squared_error
from config import CONFIG

sys.path.insert(1, str(CONFIG.src))

from utils import Visualization, Eval

vis = Visualization()
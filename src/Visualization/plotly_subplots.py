import plotly.graph_objects as go
import sys

from plotly.subplots import make_subplots
from config import CONFIG

sys.path.insert(1, str(CONFIG.data))
sys.path.insert(2, str(CONFIG.reports))



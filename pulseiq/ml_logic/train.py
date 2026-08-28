from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import numpy as np
from pulseiq.params import *


def train_model(model_class, X, y, params):
    if isinstance(params, (str, Path)):
        config_file = FRONTEND_DIR / params if isinstance(params, str) and not Path(params).is_absolute() else params
        with open(config_file, "r") as f:
            params = json.load(f)

    final_model = model_class(**params)
    final_model.fit(X, y)

    return final_model

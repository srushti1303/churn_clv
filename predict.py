from pathlib import Path
import pickle
import xgboost as xgb
from ..config import MODELS_DIR

def load_churn_model(path=None):
    p = Path(path) if path else Path(MODELS_DIR) / "churn_model.pkl"
    with open(p, "rb") as f:
        model = pickle.load(f)
    return model

def predict_churn_for_customer(model, X_customer):
    """
    X_customer: pandas DataFrame (1 row) containing features used at train time.
    Model expected to be an xgboost.Booster (trained via xgb.train).
    """
    d = xgb.DMatrix(X_customer)
    prob = model.predict(d)[0]
    return float(prob)

def load_bgf_ggf(bgf_path=None, ggf_path=None):
    bgf_p = Path(bgf_path) if bgf_path else Path(MODELS_DIR) / "bgf.pkl"
    ggf_p = Path(ggf_path) if ggf_path else Path(MODELS_DIR) / "ggf.pkl"
    with open(bgf_p, "rb") as f:
        bgf = pickle.load(f)
    with open(ggf_p, "rb") as f:
        ggf = pickle.load(f)
    return bgf, ggf

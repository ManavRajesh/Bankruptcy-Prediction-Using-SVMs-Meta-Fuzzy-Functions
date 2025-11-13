import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

def load_bank_data(csv_path: str):
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c.startswith("ratio_")]
    X = df[feature_cols].values
    y = df["failed"].values
    years = df["year"].values
    banks = df["bank"].values
    return X, y, years, banks, feature_cols, df

def time_split(df, train_years, val_year, test_year):
    fcols = [c for c in df.columns if c.startswith("ratio_")]
    tr = df[df.year.isin(train_years)]
    va = df[df.year==val_year]
    te = df[df.year==test_year]
    return (tr[fcols].values, tr["failed"].values), (va[fcols].values, va["failed"].values), (te[fcols].values, te["failed"].values)

def evaluate_scores(y_true, scores, threshold=0.0):
    # scores are margins (higher => more likely 1). Convert to preds at 0 threshold
    preds = (scores >= threshold).astype(int)
    auc = roc_auc_score(y_true, scores) if len(np.unique(y_true))==2 else float("nan")
    cm = confusion_matrix(y_true, preds, labels=[0,1])
    return {"auc": auc, "confusion_matrix": cm, "preds": preds}

def pretty_confusion(cm):
    tn, fp, fn, tp = cm.ravel()
    return f"TN:{tn} FP:{fp} FN:{fn} TP:{tp}"

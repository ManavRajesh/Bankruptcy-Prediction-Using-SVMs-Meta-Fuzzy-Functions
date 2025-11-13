import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils import load_bank_data
from src.svm_models import train_grid
from src.fuzzy_mff import cmeans_weights, apply_mff
from src.ensembles import L1_rule, L2_rule, Linf_rule
from sklearn.metrics import roc_auc_score

st.title("Bank Failure Prediction: SVM + Meta Fuzzy Ensemble")

data_path = st.text_input("Data CSV", value="data/bank_failure_dataset_realistic.csv")
train_years = st.text_input("Train years (comma)", value="1998")
val_year = st.number_input("Validation year", value=2000)
test_year = st.number_input("Test year", value=2001)
alpha = st.number_input("Alpha cutoff", value=0.0, step=0.01)

if st.button("Run"):
    X, y, years, banks, fcols, df = load_bank_data(data_path)
    tr_years = [int(s) for s in train_years.split(",") if s.strip()]
    (Xtr, ytr), (Xva, yva), (Xte, yte) = (
        df[df.year.isin(tr_years)][fcols].values, df[df.year.isin(tr_years)]["failed"].values,
        df[df.year==val_year][fcols].values, df[df.year==val_year]["failed"].values,
        df[df.year==test_year][fcols].values, df[df.year==test_year]["failed"].values,
    )
    C_list = [2**p for p in [-5,-3,-1,1,3,5]]
    sigma_list = [2**p for p in [-11,-9,-7,-5,-3,-1,1]]
    Ztr, _ = train_grid(Xtr, ytr, Xtr, C_list, sigma_list, kernel="rbf")
    Zva, _ = train_grid(Xtr, ytr, Xva, C_list, sigma_list, kernel="rbf")
    Zte, _ = train_grid(Xtr, ytr, Xte, C_list, sigma_list, kernel="rbf")
    W = cmeans_weights(Ztr, c=2, m=2.0, alpha=alpha)
    scores = apply_mff(W, Zte)[0]
    auc = roc_auc_score(yte, scores)
    st.metric("Test AUC (MFF)", f"{auc:.3f}")
    st.write("Baselines:")
    st.write({"L1": float(roc_auc_score(yte, L1_rule(Zte))),
              "L2": float(roc_auc_score(yte, L2_rule(Zte))),
              "Linf": float(roc_auc_score(yte, Linf_rule(Zte))) })

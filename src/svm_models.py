import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_svm(C, kernel="rbf", gamma=None):
    if kernel == "linear":
        clf = SVC(C=C, kernel="linear")
    else:
        clf = SVC(C=C, kernel="rbf", gamma=gamma)
    return Pipeline([("scaler", StandardScaler()), ("svc", clf)])

def train_grid(X_train, y_train, X_eval, C_list, sigma_list=None, kernel="rbf"):
    models = []
    margins = []

    # SAFETY: skip fits when only one class in y_train
    has_two_classes = len(set(y_train)) >= 2

    if kernel == "linear":
        for C in C_list:
            model = build_svm(C=C, kernel="linear")
            if has_two_classes:
                model.fit(X_train, y_train)
                m = model.decision_function(X_eval)
            else:
                m = np.zeros(len(X_eval))
            models.append((model, {"C": C, "kernel": "linear"}))
            margins.append(m)
    else:
        for C in C_list:
            for sigma in sigma_list:
                gamma = 1.0 / (2.0 * (sigma**2))
                model = build_svm(C=C, kernel="rbf", gamma=gamma)
                if has_two_classes:
                    model.fit(X_train, y_train)
                    m = model.decision_function(X_eval)
                else:
                    m = np.zeros(len(X_eval))
                models.append((model, {"C": C, "sigma": sigma, "gamma": gamma, "kernel": "rbf"}))
                margins.append(m)

    Z = np.vstack(margins) if margins else np.zeros((0, len(X_eval)))
    return Z, models

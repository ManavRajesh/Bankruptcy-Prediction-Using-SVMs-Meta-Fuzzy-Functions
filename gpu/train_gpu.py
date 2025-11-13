import argparse, json
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
try:
    from thundersvm import SVC as TsvmSVC
    HAS_TSV = True
except Exception:
    HAS_TSV = False
try:
    from cuml.svm import SVC as CuMLSVC
    HAS_CUML = True
except Exception:
    HAS_CUML = False

from src.utils import load_bank_data, time_split, evaluate_scores

def fit_decision(model, Xtr, ytr, Xev):
    model.fit(Xtr, ytr)
    # Both ThunderSVM and cuML expose decision_function as 'decision_function' or 'predict' probabilities.
    if hasattr(model, "decision_function"):
        return model.decision_function(Xev)
    else:
        # fallback to distance to hyperplane via predict + signed distances is not available -> use predict_proba if present
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(Xev)[:,1]
            return proba * 2 - 1
        return model.predict(Xev).astype(float)

def main(args):
    X, y, years, banks, fcols, df = load_bank_data(args.data)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = time_split(df, [args.train_year], args.val_year, args.test_year)
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)

    if HAS_TSV:
        SVM = TsvmSVC
        engine = "ThunderSVM"
    elif HAS_CUML:
        SVM = CuMLSVC
        engine = "cuML"
    else:
        raise RuntimeError("Neither ThunderSVM nor cuML found. Please install one of them.")

    C_list = [2**p for p in [-5,-3,-1,1,3,5]]
    sigma_list = [2**p for p in [-11,-9,-7,-5,-3,-1,1]]

    margins = []
    for C in C_list:
        for sigma in sigma_list:
            gamma = 1.0/(2.0*(sigma**2))
            model = SVM(C=C, kernel="rbf", gamma=gamma)
            margins.append(fit_decision(model, Xtr, ytr, Xte))

    import numpy as np
    Zte = np.vstack(margins)
    # simple L2 ensemble as demo
    scores = (Zte * np.abs(Zte)).sum(axis=0)
    res = evaluate_scores(yte, scores)

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    (out/"gpu_metrics.json").write_text(json.dumps({"engine":engine, "auc":res["auc"]}, indent=2))
    print("Engine:", engine, "Test AUC:", res["auc"])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/bank_failure_dataset.csv")
    p.add_argument("--train_year", type=int, default=1998)
    p.add_argument("--val_year", type=int, default=2000)
    p.add_argument("--test_year", type=int, default=2001)
    p.add_argument("--outdir", type=str, default="outputs")
    args = p.parse_args()
    main(args)

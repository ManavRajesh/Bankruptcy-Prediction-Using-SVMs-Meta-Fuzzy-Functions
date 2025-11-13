import argparse, json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.utils import load_bank_data, time_split, evaluate_scores
from src.svm_models import train_grid
# IMPORTANT: import ensemble rules here at module level
from src.ensembles import L1_rule, L2_rule, Linf_rule
from src.fuzzy_mff import cmeans_weights, apply_mff

def main(args):
    X, y, years, banks, fcols, df = load_bank_data(args.data)

    # Parse year args
    train_years = [int(y) for y in args.train_years.split(",")]
    val_year = int(args.val_year)
    test_year = int(args.test_year)

    # Time splits
    (Xtr, ytr), (Xva, yva), (Xte, yte) = time_split(df, train_years, val_year, test_year)

    # RBF grid
    C_list = [2**p for p in [-5,-3,-1,1,3,5,7,9,11,13,15]]
    sigma_list = [2**p for p in [-15,-13,-11,-9,-7,-5,-3,-1,1,3]]

    # Collect margins (TRAIN/EVAL with same trained models)
    Ztr_rbf, _ = train_grid(Xtr, ytr, Xtr, C_list, sigma_list, kernel="rbf")
    Zva_rbf, _ = train_grid(Xtr, ytr, Xva, C_list, sigma_list, kernel="rbf")
    Zte_rbf, _ = train_grid(Xtr, ytr, Xte, C_list, sigma_list, kernel="rbf")

    # Linear branch for comparison
    Ztr_lin, _ = train_grid(Xtr, ytr, Xtr, C_list, kernel="linear")
    Zva_lin, _ = train_grid(Xtr, ytr, Xva, C_list, kernel="linear")
    Zte_lin, _ = train_grid(Xtr, ytr, Xte, C_list, kernel="linear")

    # Baselines on RBF only (safe if Zte_rbf empty we’ll handle)
    def safe_auc(y_true, scores):
        try:
            return float(roc_auc_score(y_true, scores))
        except Exception:
            return float("nan")

    if Zte_rbf.size:
        l1_scores = L1_rule(Zte_rbf)
        l2_scores = L2_rule(Zte_rbf)
        linf_scores = Linf_rule(Zte_rbf)
    else:
        l1_scores = l2_scores = linf_scores = np.zeros_like(yte, dtype=float)

    # Meta Fuzzy Functions search
    best = {"auc": -1, "c": None, "m": None, "branch": None, "cluster_idx": None}
    weights_store = None

    branches = [("rbf", Ztr_rbf, Zva_rbf, Zte_rbf), ("linear", Ztr_lin, Zva_lin, Zte_lin)]
    for branch_name, Ztr, Zva, Zte in branches:
        if Ztr.size == 0 or Zva.size == 0:
            continue  # nothing to use
        for c in range(2, 5):
            for m in np.round(np.arange(1.6, 3.6, 0.2), 2):
                print(f"[INFO] Trying fuzzy clustering  c={c}  m={m}  branch={branch_name}")
                try:
                    W = cmeans_weights(Ztr, c=c, m=float(m), alpha=args.alpha)
                    val_scores = apply_mff(W, Zva)
                    aucs = []
                    for k in range(val_scores.shape[0]):
                        try:
                            a = roc_auc_score(yva, val_scores[k])
                        except Exception:
                            a = float("nan")
                        aucs.append(a)
                    print(f"     AUCs per cluster: {np.round(aucs,3)}")
                    k_best = int(np.nanargmax(aucs))
                    if np.nanmax(aucs) > best["auc"]:
                        best.update({"auc": float(np.nanmax(aucs)), "c": c, "m": float(m),
                                    "branch": branch_name, "cluster_idx": k_best})
                        weights_store = (branch_name, W)
                        print(f"     🔹 New best AUC={best['auc']}  at c={c}, m={m}")
                except Exception as e:
                    print(f"     ⚠️  Fuzzy clustering failed for c={c}, m={m}: {e}")

                except Exception:
                    # skip bad combos
                    continue

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # If MFF search failed, fall back gracefully
    if weights_store is None:
        fallback_auc = safe_auc(yte, l2_scores)
        (outdir/"metrics.json").write_text(json.dumps({
            "best_mff": None,
            "test_auc": fallback_auc,
            "note": "Fell back to L2 ensemble because no valid fuzzy weights found."
        }, indent=2))
        (outdir/"baselines.json").write_text(json.dumps({
            "L1_auc": safe_auc(yte, l1_scores),
            "L2_auc": fallback_auc,
            "Linf_auc": safe_auc(yte, linf_scores)
        }, indent=2))
        print("⚠️ No valid MFF found. Fallback L2 AUC:", fallback_auc)
        return

    # Evaluate with best MFF on corresponding branch
    branch_name, W = weights_store
    Zte = Zte_rbf if branch_name == "rbf" else Zte_lin
    if Zte.size == 0:
        test_auc = float("nan")
        cm = [[0,0],[0,0]]
        scores = np.zeros_like(yte, dtype=float)
    else:
        scores = apply_mff(W, Zte)[best["cluster_idx"]]
        res = evaluate_scores(yte, scores)
        test_auc = float(res["auc"])
        cm = res["confusion_matrix"].tolist()
        # Save fuzzy weight matrix for visualization
    np.save(outdir / "last_weights.npy", W)


    # Save results
    (outdir/"metrics.json").write_text(json.dumps({
        "best_mff": best,
        "test_auc": test_auc,
        "confusion_matrix": cm
    }, indent=2))
    (outdir/"baselines.json").write_text(json.dumps({
        "L1_auc": safe_auc(yte, l1_scores),
        "L2_auc": safe_auc(yte, l2_scores),
        "Linf_auc": safe_auc(yte, linf_scores)
    }, indent=2))

    print("✅ Best MFF:", best)
    print("✅ Test AUC:", test_auc)
    print("✅ Baselines  L1:", safe_auc(yte, l1_scores),
          " L2:", safe_auc(yte, l2_scores),
          " Linf:", safe_auc(yte, linf_scores))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/bank_failure_dataset_realistic.csv")
    p.add_argument("--train_years", type=str, default="1998,1999,2000,2001")
    p.add_argument("--val_year", type=int, default=2001)
    p.add_argument("--test_year", type=int, default=2001)
    p.add_argument("--alpha", type=float, default=0.0, help="alpha cutoff for memberships")
    p.add_argument("--outdir", type=str, default="outputs")
    args = p.parse_args()
    main(args)

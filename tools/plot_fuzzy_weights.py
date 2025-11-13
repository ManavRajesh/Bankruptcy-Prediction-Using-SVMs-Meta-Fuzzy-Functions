import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def plot_fuzzy_weights(W, title="Fuzzy Membership Matrix (W)", savepath=None):
    W = np.array(W, dtype=float)

    print("W shape:", W.shape)
    print("min:", W.min(), "max:", W.max())

    plt.figure(figsize=(12, 6))

    sns.heatmap(
        W,
        cmap="magma",
        cbar=True,
        square=False
    )

    plt.title(title)
    plt.xlabel("SVM Model Index")
    plt.ylabel("Cluster Index")
    plt.tight_layout()

    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=300)

    plt.show()
    plt.close()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default="outputs/last_weights.npy")
    args = p.parse_args()

    W = np.load(args.weights)
    plot_fuzzy_weights(W)
    plt.savefig("outputs/fuzzy_heatmap.png", dpi=300)
    print("✅ Saved heatmap to outputs/fuzzy_heatmap.png")
    plt.close()
    plt.savefig("outputs/fuzzy_heatmap.png", dpi=300)
    print("✅ Saved heatmap to outputs/fuzzy_heatmap.png")
    plt.close()

import numpy as np
import skfuzzy as fuzz

def cmeans_weights(Ztr, c=3, m=2.0, alpha=0.0, error=0.005, maxiter=1000):
    """
    Perform fuzzy c-means clustering on model outputs (Ztr).
    Each model is treated as a data point in feature space defined by its margins.

    Parameters
    ----------
    Ztr : ndarray (models × samples)
        Decision margin matrix from base learners.
    c : int
        Number of fuzzy clusters.
    m : float
        Fuzziness coefficient.
    alpha : float
        Alpha-cut threshold (to zero out small memberships).
    error : float
        Convergence tolerance.
    maxiter : int
        Maximum iterations.

    Returns
    -------
    W : ndarray (clusters × models)
        Fuzzy membership weights for each cluster.
    """

    Ztr = np.array(Ztr)
    if Ztr.ndim == 1:
        Ztr = Ztr.reshape(1, -1)

    # Transpose so each model = sample, each feature = decision value
    data = Ztr  # models × samples
    data_T = data  # shape: models × samples

    # Fuzzy C-means expects features × samples
    # so transpose to samples along columns
    result = fuzz.cluster.cmeans(data_T.T, c=c, m=m, error=error, maxiter=maxiter)

    # Handle scikit-fuzzy versions (old/new)
    if len(result) == 3:
        U, U0, d = result
    elif len(result) == 4:
        cntr, U, U0, d = result
    elif len(result) == 7:
        cntr, U, U0, d, jm, p, fpc = result
    else:
        raise ValueError(f"Unexpected number of values returned from cmeans(): {len(result)}")

    # U shape: (clusters × samples) = (c × models)
    W = np.copy(U)

    # Optional alpha cut
    if alpha > 0.0:
        W[W < alpha] = 0.0

    # Normalize rows to sum to 1
    W = W / (W.sum(axis=1, keepdims=True) + 1e-8)
    return W


def apply_mff(W, Z):
    """
    Apply Meta-Fuzzy Function ensemble weighting to model outputs.

    Parameters
    ----------
    W : ndarray (clusters × models)
        Fuzzy weights per cluster.
    Z : ndarray (models × samples)
        Model outputs.

    Returns
    -------
    F : ndarray (clusters × samples)
        Fuzzy-aggregated ensemble scores.
    """
    Z = np.array(Z)
    if Z.ndim == 1:
        Z = Z.reshape(1, -1)
    if W.ndim == 1:
        W = W.reshape(1, -1)

    # Ensure proper orientation
    if W.shape[1] != Z.shape[0]:
        raise ValueError(f"Shape mismatch: W={W.shape}, Z={Z.shape}. Expected W second dim = Z first dim.")

    F = np.dot(W, Z)
    return F

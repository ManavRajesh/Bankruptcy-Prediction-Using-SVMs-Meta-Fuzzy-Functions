import numpy as np

def L1_rule(Z):
    """Sum of margins across models"""
    return Z.sum(axis=0)

def L2_rule(Z):
    """Weighted by absolute margin"""
    return (Z * np.abs(Z)).sum(axis=0)

def Linf_rule(Z):
    """max positive + min negative margin per sample"""
    max_pos = np.max(np.where(Z > 0, Z, -np.inf), axis=0)
    min_neg = np.min(np.where(Z < 0, Z,  np.inf), axis=0)
    max_pos = np.where(np.isfinite(max_pos), max_pos, 0.0)
    min_neg = np.where(np.isfinite(min_neg), min_neg, 0.0)
    return max_pos + min_neg

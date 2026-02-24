"""
NetworkBuilder (GPU - PyTorch)

Costruisce una rete gene x gene unificata a partire dalle quattro viste ontologiche (BP, CC, MF, HPO),
implementando un'accelerazione hardware custom (CUDA/MPS) tramite PyTorch per risolvere i colli di bottiglia computazionali.

1. MATRICI DI AFFINITÀ CUSTOM (_cosine_distance, _affinity_matrix):
   - Calcolo ottimizzato della distanza coseno tramite operazioni matriciali su tensori GPU.
   - Costruzione del kernel esponenziale scalato (KNN-based) equivalente all'implementazione snfpy.

2. SNF — SIMILARITY NETWORK FUSION GPU (_snf_fuse):
   - Fusione iterativa delle matrici di affinità implementata interamente su tensori PyTorch.
   - Utilizza i "dominant sets" (KNN-thresholding) per sparsificare il rumore ad ogni iterazione.

3. GESTIONE DEVICE E SALVATAGGIO:
   - Rilevamento automatico dell'hardware disponibile (MPS per Mac Apple Silicon, CUDA per Nvidia, con fallback CPU).
   - Ritorno in memoria CPU (NumPy/Pandas) e salvataggio della matrice fusa finale in formato CSV.
"""

import numpy as np
import pandas as pd
import torch
import os
from typing import List


def _get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _cosine_distance(X: torch.Tensor) -> torch.Tensor:
    """Cosine distance via a single GEMM: D = 1 - norm(X) @ norm(X).T"""
    X_norm = X / X.norm(dim=1, keepdim=True).clamp(min=1e-12)
    sim = X_norm @ X_norm.T
    return (1.0 - sim).clamp(min=0.0)


def _affinity_matrix(dist: torch.Tensor, K: int = 20, mu: float = 0.5) -> torch.Tensor:
    """Scaled exponential similarity kernel (matches snfpy's affinity_matrix)."""
    N = dist.shape[0]
    dist = dist.clone()
    dist.fill_diagonal_(0.0)

    T_sorted, _ = dist.sort(dim=1)
    # Mean distance to K nearest neighbors (exclude self at col 0)
    TT = T_sorted[:, 1:K + 1].mean(dim=1, keepdim=True)

    sigma = (TT + TT.T + dist) / 3.0
    sigma = sigma.clamp(min=1e-12)

    scale = mu * sigma
    # Gaussian kernel: exp(-d² / 2σ²) / (σ√2π)
    W = torch.exp(-dist ** 2 / (2 * scale ** 2)) / (scale * np.sqrt(2 * np.pi))
    W = (W + W.T) / 2.0
    W.fill_diagonal_(0.0)
    return W


def _dominant_set(W: torch.Tensor, K: int = 20) -> torch.Tensor:
    """KNN-thresholded row-normalized matrix (matches snfpy's _find_dominate_set)."""
    N = W.shape[0]
    Wk = W.clone()
    # Keep only top-K entries per row
    k_keep = max(1, int(K))
    if k_keep < N:
        cutoff_idx = N - k_keep
        vals_sorted, _ = Wk.sort(dim=1)
        cutoffs = vals_sorted[:, cutoff_idx:cutoff_idx + 1]
        Wk[Wk < cutoffs] = 0.0
    row_sums = Wk.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return Wk / row_sums


def _snf_fuse(affinities: List[torch.Tensor], K: int = 20, t: int = 20, alpha: float = 1.0) -> torch.Tensor:
    """SNF iterative fusion on GPU tensors."""
    m = len(affinities)
    N = affinities[0].shape[0]
    eye = torch.eye(N, device=affinities[0].device, dtype=affinities[0].dtype)

    # Row-normalize and symmetrize
    aff = []
    for mat in affinities:
        mat = mat / mat.sum(dim=1, keepdim=True).clamp(min=1e-12)
        mat = (mat + mat.T) / 2.0
        aff.append(mat)

    # Build KNN sparse matrices
    Wk = [_dominant_set(a, K) for a in aff]
    Wsum = sum(aff)

    for _ in range(t):
        for n in range(m):
            nzW = Wk[n]
            # Core SNF update: S^(v) @ (Σ P^(k≠v) / (m-1)) @ S^(v).T
            aff[n] = nzW @ (Wsum - aff[n]) @ nzW.T / (m - 1)
            aff[n] = aff[n] + alpha * eye
            aff[n] = (aff[n] + aff[n].T) / 2.0
        Wsum = sum(aff)

    W = Wsum / m
    W = W / W.sum(dim=1, keepdim=True).clamp(min=1e-12)
    W = (W + W.T + eye) / 2.0
    return W


class NetworkBuilder:
    def __init__(self, bp, cc, mf, hpo, output_dir="../data/processed/"):
        self.views = {"BP": bp, "CC": cc, "MF": mf, "HPO": hpo}
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def build_and_fuse(self, K=20, mu=0.5, t=20):
        """
        Build per-view affinity matrices from TF-IDF features using cosine
        distance, then fuse them with SNF.

        Uses GPU (MPS/CUDA) when available, falls back to CPU.
        Returns the fused (N, N) similarity matrix as a DataFrame.
        """
        device = _get_device()
        print(f"=== Similarity Network Fusion (device: {device}) ===\n")

        gene_index = list(self.views.values())[0].index

        # Build affinity matrices on GPU
        affinities = []
        for name, df in self.views.items():
            X = torch.tensor(df.values, dtype=torch.float32, device=device)
            print(f"[{name}] Feature matrix: {tuple(X.shape)}")

            dist = _cosine_distance(X)
            aff = _affinity_matrix(dist, K=K, mu=mu)
            affinities.append(aff)
            del X, dist  # free intermediate memory

        print(f"\n[SNF] Fusing {len(affinities)} networks (K={K}, t={t} iterations)...")
        fused = _snf_fuse(affinities, K=K, t=t)

        # Move back to CPU / numpy
        fused_np = fused.cpu().numpy().astype(np.float64)
        fused_df = pd.DataFrame(fused_np, index=gene_index, columns=gene_index)

        fused_df.to_csv(f"{self.output_dir}fused_network.csv")
        print(f"[OK] Fused network: {fused_df.shape}")
        print(f"[SAVED] {self.output_dir}fused_network.csv")

        return fused_df

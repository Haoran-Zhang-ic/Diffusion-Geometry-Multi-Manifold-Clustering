import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
from sklearn.cluster import SpectralClustering
from ManifoldDiffusionGeometry import Tangents, Dimension_Estimate
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import lil_matrix
from tqdm import tqdm
from sklearn.mixture import GaussianMixture


def compute_pij(tangent_bundle_d, i, j, o=8):
    Ui = tangent_bundle_d[i]
    Uj = tangent_bundle_d[j]
    M = Ui.T @ Uj
    _, S, _ = np.linalg.svd(M)
    cosines = np.clip(S, -1.0, 1.0)
    pij = np.prod(cosines) ** o
    return pij


def partitioning(X, M=10, max_iter=100, random_state=None):
    gm = GaussianMixture(n_components=M, covariance_type='full',
                         max_iter=max_iter, random_state=random_state)
    gm.fit(X)
    labels = gm.predict(X)
    return labels


def clustering(partition_labels, pre_n_class, X, n_class, K, o=8, random_state=None):
    stages = ["Tangent Space Estimation", "Dimension Estimation", "Constructing Affinity Matrix", "Spectral Clustering"]
    with tqdm(total=len(stages), desc="Process") as pbar:
        tangent_bundles = []
        tangent_idx = []
        for label in range(pre_n_class):
            if X[partition_labels == label, :].shape[0] == 0:
                continue
            elif X[partition_labels == label, :].shape[0] <= 2:
                tangent_bundle, pointwise_eigenvalues, L, Gamma, data_bandlimited, u, sample_density = Tangents(X,
                                                                                                                return_all=True)
                tangent_bundles.append(tangent_bundle[partition_labels == label])
                tangent_idx.append(np.arange(X.shape[0])[partition_labels == label])
            else:
                tangent_bundle, pointwise_eigenvalues, L, Gamma, data_bandlimited, u, sample_density = Tangents(
                    X[partition_labels == label, :], return_all=True)
                tangent_bundles.append(tangent_bundle)
                tangent_idx.append(np.arange(X.shape[0])[partition_labels == label])
        pbar.set_description(stages[0])
        pbar.update(1)
        tangent_bundle_tot = np.zeros((X.shape[0], X.shape[1], X.shape[1]))
        for i in range(len(tangent_idx)):
            tangent_bundle_tot[tangent_idx[i]] = tangent_bundles[i]
        d, _ = Dimension_Estimate(X)
        pbar.set_description(stages[1])
        pbar.update(1)
        tangent_bundle_tot = tangent_bundle_tot[:, :, :d]
        N = X.shape[0]
        nbrs = NearestNeighbors(n_neighbors=K, algorithm='auto').fit(X)
        _, nbr_indices = nbrs.kneighbors(X)

        W = lil_matrix((N, N))

        neighbor_pairs = set()
        for i in range(N):
            for j in nbr_indices[i]:
                pair = tuple(sorted((i, j)))
                neighbor_pairs.add(pair)

        for i, j in neighbor_pairs:
            pij = compute_pij(tangent_bundle_tot, i, j, o=o)
            W[i, j] = pij
            W[j, i] = pij
        pbar.set_description(stages[2])
        pbar.update(1)
        W = W.toarray()
        np.random.seed(random_state)
        sc = SpectralClustering(
            n_clusters=n_class,
            affinity='precomputed',
            assign_labels='discretize',
            random_state=random_state
        )
        labels = sc.fit_predict(W)
        pbar.set_description(stages[3])
        pbar.update(1)
    return labels


def DiffusionGeometrySpectralClustering(X, n_class, M=10, K=60, o=8, random_state=None):
    partition_labels = partitioning(X, M=M, random_state=random_state)
    labels = clustering(partition_labels, M, X, n_class, K, o=o, random_state=random_state)
    return labels

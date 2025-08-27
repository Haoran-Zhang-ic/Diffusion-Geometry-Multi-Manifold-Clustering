import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
from opt_einsum import contract
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import diags, coo_matrix, eye
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import scipy as sp


def Del0_adaptive_operator(x, n0=60, c=0, k=32, k_ep=8):
    n = x.shape[0]

    # 限制 k
    k = min(k, max(2, int(0.2 * n)))  # 0.2
    # 限制 k_ep
    k_ep = min(k_ep, max(2, k // 3))  # 3

    # 限制 n0
    n0 = min(n0, n - 1)  # eigsh 要求 n0 < n
    ## 1. Setup

    # Find k nearest neighbours
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(x)
    d, nbr_indices = nbrs.kneighbors(x)

    # Compute the ad hoc bandwidth function rho0
    rho0 = np.sqrt((d[:, 1:k_ep] ** 2).mean(axis=1))

    ## 2. Tune the initial paramters epsilon0 and dim0 to use for density estimation.

    # Compute the pre-kernel K0 with rho0 over a range of possible epsilons.
    epsilons = 2 ** np.arange(-30, 10, 0.25)
    d0 = d ** 2 / (rho0[nbr_indices] * rho0.reshape(-1, 1))
    K0 = np.exp(-d0.reshape(n, k, 1) / (2 * epsilons.reshape(1, 1, -1)))
    K0_global = K0.mean(axis=(0, 1))

    # Select the epsilon0 that maximises the criterion and estimate dim0.
    criterion0 = np.diff(np.log(K0_global)) / np.diff(np.log(epsilons))
    max_index0 = np.argmax(criterion0)
    epsilon0 = epsilons[max_index0]
    dim0 = 2 * criterion0[max_index0]
    # print(dim0, epsilon0)

    ## 3. Use epsilon0 and dim0 to estimate the density qest.

    # Compute the heat kernel matrix and symmetrise.
    K1 = np.exp(-d0 / (2 * epsilon0)) / ((2 * np.pi * epsilon0) ** (dim0 / 2))
    K1 = coo_matrix((K1.flatten(), (np.repeat(np.arange(n), k), nbr_indices.flatten())))
    K1 = (K1 + K1.T) / 2

    # Sum over the rows to get a density estimate.
    qest = K1.sum(axis=0) / (n * rho0 ** dim0)
    qest = qest.A[0]

    ## 4. Define the true bandwidth function rho with the density estimate qest.

    # Set alpha and beta in terms of the parameter c.
    alpha = 1 / 2 - c / 2 - dim0 / 4  #### dim or dim0
    beta = -1 / 2

    # Define rho with qest.
    rho = qest ** beta
    rho /= np.median(rho)
    # rho /= np.mean(rho)

    ## 5. Tune the final paramters epsilon and dim.

    # Compute the kernel K2 with rho0 over a range of possible epsilons.
    d2 = d ** 2 / (rho[nbr_indices] * rho.reshape(-1, 1))
    K2 = np.exp(-d2.reshape(n, k, 1) / (4 * epsilons.reshape(1, 1, -1)))
    K2_global = K2.mean(axis=(0, 1))

    # Select the epsilon that maximises the criterion and estimate dim.
    criterion = np.diff(np.log(K2_global)) / np.diff(np.log(epsilons))
    # px.scatter(criterion).show()
    max_index = np.argmax(criterion)
    epsilon = epsilons[max_index]
    dim = 2 * criterion[max_index]
    # dim0=dim

    ## 6. Define the final kernel matrix K and alpha-normalise.

    # Compute K with the final epsilon and dim (K is K_ep).
    K = np.exp(-d2 / (4 * epsilon))
    K = coo_matrix((K.flatten(), (np.repeat(np.arange(n), k), nbr_indices.flatten())))
    K = (K + K.T) / 2

    # Normalise K with the 'alpha' normalisation (qest is q_ep, K becomes K_{ep,al}).
    qest = K.sum(axis=0).A[0] / (rho ** dim)  #### dim or dim0?
    alpha_normalisation = diags(qest ** (-alpha))
    K = alpha_normalisation @ K @ alpha_normalisation

    ## 7. Solve the eigenproblem for K via symmetric normalisation.

    # Compute the normalisations D and P.
    D = K.sum(axis=0).A[0]
    P2 = rho ** 2

    # Compute the Laplacian as an operator.
    L = diags(1 / P2) @ (eye(n) - diags(1 / D) @ K) / epsilon
    # Form the symmetric normalisation of K.
    sample_density = D * P2
    S_normalisation = diags(sample_density ** (-1 / 2))
    K = S_normalisation @ K @ S_normalisation - diags(1 / P2)
    K /= epsilon

    # Find eigenvalues and eigenfunctions of K, un-normalise, and clean up data.
    v0 = np.ones(K.shape[0], dtype=float)
    lam, u = eigsh(K, n0, which='LA', v0=v0)
    u = S_normalisation @ u
    u = u[:, ::-1].T

    return L, u, sample_density


def pointwise_decomp(pointwise_matrices):
    vals, vecs = [], []
    for m in pointwise_matrices:
        a, b = eigh(m)
        vals.append(a[::-1])
        vecs.append(b[:, ::-1])
    return np.array(vals), np.array(vecs)


def Tangents(data, n0=100, return_all=False):
    n = data.shape[0]

    # Compute the Laplacian.
    L, u, sample_density = Del0_adaptive_operator(data, n0=n0)

    # Bandlimit the data to the first n0 eigenfunctions (i.e. smooth it a bit).
    data_bandlimited = u.T @ u * sample_density @ data

    # Compute the carré du champ (Gamma) pointwise, i.e. metric in the tangent space for each x.
    XiLXj = data_bandlimited.reshape(n, -1, 1) * (L @ data_bandlimited).reshape(n, 1, -1)
    coord_products = (data_bandlimited.reshape(n, -1, 1) * data_bandlimited.reshape(n, 1, -1))
    LXiXj = (L @ coord_products.reshape(n, -1)).reshape(coord_products.shape)
    Gamma = (1 / 2) * (XiLXj + XiLXj.transpose((0, 2, 1)) - LXiXj)

    # Bandlimit Gamma to the first n0 eigenfunctions.
    Gamma_bandlimited = (u.T @ u * sample_density @ Gamma.reshape(n, -1)).reshape(Gamma.shape)

    # Diagonalise Gamma at each point to get an orthornomal basis for each tangent space.
    pointwise_eigenvalues, tangent_bundle = pointwise_decomp(Gamma_bandlimited)
    # pointwise_norms = contract('pij,pik,pjk->pk',Gamma,tangent_bundle,tangent_bundle)

    if return_all:
        return tangent_bundle, pointwise_eigenvalues, L, Gamma, data_bandlimited, u, sample_density

    else:
        return tangent_bundle


def Dimension_Estimate(data, n0=80):
    # Compute eigenvalues of the metric at each point, as well as the
    # eigenfunctions of the Laplacian and sample density.
    _, pointwise_eigenvalues, _, _, _, u, sample_density = Tangents(data, return_all=True,
                                                                    n0=min(data.shape[0] - 1, n0))

    # Estimate the pointwise dimension as the 'elbow' in the metric eigenvalues.
    pointwise_differences = - np.diff(pointwise_eigenvalues, prepend=1, append=0)
    pointwise_dimensions = np.argmax(pointwise_differences, axis=1)

    # Bandlimit this pointwise estimate.
    pointwise_dimensions = u.T @ u * sample_density @ pointwise_dimensions

    # Estimate the global dimension as the median pointwise dimension.
    global_dimension = np.median(pointwise_dimensions).round()
    return max(int(global_dimension), 1), pointwise_dimensions.round()

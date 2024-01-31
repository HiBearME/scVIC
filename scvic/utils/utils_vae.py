from scvi.inference import Posterior
import scanpy as sc
import numpy as np
from sklearn.mixture import GaussianMixture
import logging

logger = logging.getLogger(__name__)


def Louvain(
        posterior: Posterior,
        n_neighbors=15,
        resolution=1.0
):

    latent, _, _ = posterior.get_latent()
    n_samples = latent.shape[0]
    sc_latent = sc.AnnData(latent)
    if n_samples > 200000:
        log_message = "Number of samples is quit large, " \
                      "resample 20,0000 samples to estimate parameters"
        logger.info(log_message)
        sc_latent = sc_latent[np.random.choice(n_samples, 200000, replace=False)]

    log_message = "Construct KNN graph before Louvain in scanpy"
    logger.info(log_message)
    sc.pp.neighbors(sc_latent, n_neighbors=n_neighbors, use_rep="X")

    log_message = "Run Louvain"
    logger.info(log_message)
    sc.tl.louvain(sc_latent, resolution=resolution)
    louvain_labels = sc_latent.obs['louvain']
    louvain_labels = np.asarray(louvain_labels, dtype=int)
    n_clusters = np.unique(louvain_labels).shape[0]
    if n_clusters <= 1:
        exit("Error: There is only a cluster detected. The resolution:" + str(
            resolution) + "is too small, choose a larger resolution.")
    ratio = []
    mu = []
    for i in range(n_clusters):
        indices = louvain_labels == i
        ratio.append(sum(indices))
        mu.append(latent[indices].mean(axis=0).reshape(1, -1))
    ratio = np.array(ratio) / n_samples
    mu = np.concatenate(mu)
    return mu, ratio, latent, louvain_labels


def GMM(
        posterior: Posterior,
        n_components,
        covariance_type='full'
):

    latent, _, _ = posterior.get_latent()
    n_samples = latent.shape[0]
    if n_samples > 200000:
        log_message = "Number of samples is quit large, " \
                      "resample 20,0000 samples to estimate parameters"
        logger.info(log_message)
        latent = latent[np.random.choice(n_samples, 200000, replace=False)]
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type)

    log_message = "Fit gaussian mixture model"
    logger.info(log_message)
    gmm.fit(latent)
    labels = gmm.predict(latent)
    return gmm.means_, gmm.weights_, latent, labels

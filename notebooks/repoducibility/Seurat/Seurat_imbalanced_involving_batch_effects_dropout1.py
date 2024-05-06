import random
import numpy as np
import pandas as pd

seed = 1

from sklearn.neighbors import NearestNeighbors
from scipy.sparse import identity
from scipy.stats import entropy

def entropy_batch_mixing(
    latent_space, batches, n_batches, n_neighbors=50, n_pools=50, n_samples_per_pool=100
):
    def counts_frequency(hist_data):
        counts = np.zeros(n_batches)
        for batch in range(n_batches):
            counts[batch] = sum(hist_data == batch)
        return np.array(counts) / sum(counts)

    frequency_prior = counts_frequency(batches)

    def kl_divergence(hist_data):
        frequency_posterior = counts_frequency(hist_data)
        return entropy(frequency_posterior, frequency_prior)

    n_neighbors = min(n_neighbors, len(latent_space) - 1)
    nne = NearestNeighbors(n_neighbors=1 + n_neighbors, n_jobs=4)
    nne.fit(latent_space)
    kmatrix = nne.kneighbors_graph(latent_space) - identity(latent_space.shape[0])

    score = 0
    for t in range(n_pools):
        indices = np.random.choice(
            np.arange(latent_space.shape[0]), size=n_samples_per_pool
        )
        score += np.mean(
            [
                kl_divergence(
                    batches[
                        kmatrix[indices].nonzero()[1][
                            kmatrix[indices].nonzero()[0] == i
                        ]
                    ]
                )
                for i in range(n_samples_per_pool)
            ]
        )
    return score / float(n_pools)

def getNClusters(data, n_cluster, range_min=0, range_max=4, max_steps=20, verbose=False):
    if isinstance(data, np.ndarray):
        data = anndata.AnnData(X=data)
    assert isinstance(data, anndata.AnnData), \
        "data must be numpy.ndarray or anndata.AnnData"
    this_min = range_min
    this_max = range_max
    for this_step in range(max_steps):
        if verbose:
            print("step: " + str(this_step))
        this_resolution = this_min + ((this_max-this_min)/2)
        sc.pp.neighbors(data)
        sc.tl.louvain(data, resolution=this_resolution)
        this_clusters = len(np.unique(data.obs['louvain']))
        if verbose:
            print('got ' + str(this_clusters) + ' at resolution ' + str(this_resolution))
        if this_clusters > n_cluster:
            this_max = this_resolution
        if this_clusters < n_cluster:
            this_min = this_resolution
        if this_clusters == n_cluster:
            if not verbose:
                print('got ' + str(this_clusters) + ' at resolution ' + str(this_resolution))
            return data
    print('Cannot find the number of clusters')

import scanpy as sc
import anndata
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scvic.utils import cluster_acc


i_dropout = 0

ARI_Seurat_Louvain = np.zeros((5, 10))
NMI_Seurat_Louvain = np.zeros((5, 10))
ACC_Seurat_Louvain = np.zeros((5, 10))
BatchMixing_Seurat_Louvain = np.zeros((5, 10))

for i_ratio in range(5):
    for i_duplicate in range(10):
        
        random.seed(seed)
        np.random.seed(seed)
        
        groups = pd.read_csv(
        "/Users/Healthy/Desktop/final/R/generate simulated datasets/data/Imbalanced Datasets Involving Batch Effects/ratio" + str(i_ratio + 1) + "/dropout" + str(i_dropout + 1) +"/groups_duplicate" + str(i_duplicate + 1) + ".csv",
                    index_col=0).values.squeeze()
        batches = pd.read_csv(
        "/Users/Healthy/Desktop/final/R/generate simulated datasets/data/Imbalanced Datasets Involving Batch Effects/ratio" + str(i_ratio + 1) + "/dropout" + str(i_dropout + 1) +"/batches_duplicate" + str(i_duplicate + 1) + ".csv",
                    index_col=0).values.squeeze()
        pca = pd.read_csv(
        "./intermediate_saving/pcs/Imbalanced Datasets Involving Batch Effects/ratio" + str(i_ratio + 1) + "/dropout" + str(i_dropout + 1) +"/pcs_duplicate" + str(i_duplicate + 1) + ".csv", index_col=0).values
                
        cell_types = np.unique(groups).tolist()
        n_labels = len(cell_types)
        labels = []
        for i in range(len(groups)):
            for j in range(len(cell_types)):
                if groups[i] == cell_types[j]:
                    labels.append(j)
                    continue
        
        batch_types = np.unique(batches).tolist()
        batch_indices = []
        n_batches = len(batch_types)
        for i in range(len(batches)):
            for j in range(len(batch_types)):
                if batches[i] == batch_types[j]:
                    batch_indices.append(j)
                    continue
                    
                        
        data = getNClusters(pca, n_cluster=n_labels)
        ARI_Seurat_Louvain[i_ratio, i_duplicate] = np.around(adjusted_rand_score(labels, data.obs["louvain"]), 5)
        NMI_Seurat_Louvain[i_ratio, i_duplicate] = np.around(normalized_mutual_info_score(labels, data.obs["louvain"], average_method='arithmetic'), 5)
        ACC_Seurat_Louvain[i_ratio, i_duplicate] = np.around(cluster_acc(labels, data.obs["louvain"].astype(int)), 5)                                                                
        BatchMixing_Seurat_Louvain[i_ratio, i_duplicate] = np.around(entropy_batch_mixing(pca, np.array(batch_indices), n_batches), 5)

print("ARI_Seurat_Louvain: ")
print(ARI_Seurat_Louvain)
np.save("saved/ARI_Seurat_Louvain_imbalanced_involving_batch_effects_dropout1.npy", ARI_Seurat_Louvain)

print("NMI_Seurat_Louvain: ")
print(NMI_Seurat_Louvain)
np.save("saved/NMI_Seurat_Louvain_imbalanced_involving_batch_effects_dropout1.npy", NMI_Seurat_Louvain)

print("ACC_Seurat_Louvain: ")
print(ACC_Seurat_Louvain)
np.save("saved/ACC_Seurat_Louvain_imbalanced_involving_batch_effects_dropout1.npy", ACC_Seurat_Louvain)     

print("BatchMixing_Seurat_Louvain: ")
print(BatchMixing_Seurat_Louvain)
np.save("saved/BatchMixing_Seurat_Louvain_imbalanced_involving_batch_effects_dropout1.npy", BatchMixing_Seurat_Louvain)
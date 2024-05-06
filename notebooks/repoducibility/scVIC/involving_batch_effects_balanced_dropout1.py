import random
import pandas as pd
import numpy as np
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

seed = 1

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

from scvic.dataset import ExpressionDataset
import scanpy as sc
import anndata
from scvic.models import CVAE
from scvic.inference import CTrainer
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scvic.utils import cluster_acc, entropy_batch_mixing


n_epochs = 800
lr = 1e-2
use_cuda = True
n_gene_highly_variable = 500

i_dropout = 0

ARI_scVIC = np.zeros((5, 10))
ARI_scVIC_Louvain = np.zeros((5, 10))
NMI_scVIC = np.zeros((5, 10))
NMI_scVIC_Louvain = np.zeros((5, 10))
ACC_scVIC = np.zeros((5, 10))
ACC_scVIC_Louvain = np.zeros((5, 10))
BatchMixing_scVIC = np.zeros((5, 10))
BatchMixing_scVIC_Louvain = np.zeros((5, 10))

for i_cluster in range(5):
    for i_duplicate in range(10):
        
        print("i_cluster: ", i_cluster, ", i_duplicate: ", i_duplicate)
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        
        counts = pd.read_csv(
        "/Users/Healthy/Desktop/final/R/generate simulated datasets/data/Balanced Datasets Involving Batch Effects/" + str(i_cluster + 3) + "/dropout" + str(i_dropout + 1) +"/counts_duplicate" + str(i_duplicate + 1) + ".csv",
                    index_col=0).values.T
        groups = pd.read_csv(
        "/Users/Healthy/Desktop/final/R/generate simulated datasets/data/Balanced Datasets Involving Batch Effects/" + str(i_cluster + 3) + "/dropout" + str(i_dropout + 1) +"/groups_duplicate" + str(i_duplicate + 1) + ".csv",
                    index_col=0).values.squeeze()
        batches = pd.read_csv(
        "/Users/Healthy/Desktop/final/R/generate simulated datasets/data/Balanced Datasets Involving Batch Effects/" + str(i_cluster + 3) + "/dropout" + str(i_dropout + 1) +"/batches_duplicate" + str(i_duplicate + 1) + ".csv",
                    index_col=0).values.squeeze()
                
        cell_types = np.unique(groups).tolist()
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
                    
        gene_dataset = ExpressionDataset()
        gene_dataset.populate_from_data(X=counts, labels=labels, batch_indices=batch_indices, cell_types=cell_types)
        gene_dataset.keep_highly_variable_genes_by_seurat(new_n_genes=500, batch_correction=True)
        n_labels = gene_dataset.n_labels
    
        cvae = CVAE(gene_dataset.nb_genes, n_labels=n_labels, n_batch=n_batches, n_latent=10)
        ctrainer = CTrainer(
            cvae,
            gene_dataset,
            train_size=1.0,
            use_cuda=use_cuda,
            n_epochs_kl_warmup=None,
            n_epochs_pre_train=400
        )
                
        ctrainer.train(n_epochs=n_epochs, lr=lr)
        ctrainer.more_steps_for_gmm(max_steps=400)
        full = ctrainer.create_posterior(ctrainer.model, gene_dataset, indices=np.arange(len(gene_dataset)))
        latent, labels_pred = full.sequential().get_latent()
                
        ARI_scVIC[i_cluster, i_duplicate] = np.around(adjusted_rand_score(gene_dataset.labels.squeeze(), labels_pred), 5)
        NMI_scVIC[i_cluster, i_duplicate] = np.around(normalized_mutual_info_score(gene_dataset.labels.squeeze(), labels_pred, average_method='arithmetic'), 5)
        ACC_scVIC[i_cluster, i_duplicate] = np.around(cluster_acc(gene_dataset.labels.squeeze(), labels_pred), 5)
        BatchMixing_scVIC[i_cluster, i_duplicate] = np.around(entropy_batch_mixing(latent, gene_dataset.batch_indices, gene_dataset.n_batches), 5)                                                 
        
        data = getNClusters(latent, n_cluster=n_labels)
        ARI_scVIC_Louvain[i_cluster, i_duplicate] = np.around(adjusted_rand_score(gene_dataset.labels.squeeze(), data.obs["louvain"]), 5)
        NMI_scVIC_Louvain[i_cluster, i_duplicate] = np.around(normalized_mutual_info_score(gene_dataset.labels.squeeze(), data.obs["louvain"], average_method='arithmetic'), 5)
        ACC_scVIC_Louvain[i_cluster, i_duplicate] = np.around(cluster_acc(gene_dataset.labels.squeeze(), data.obs["louvain"].astype(int)), 5)
        BatchMixing_scVIC_Louvain[i_cluster, i_duplicate] = np.around(entropy_batch_mixing(latent, gene_dataset.batch_indices, gene_dataset.n_batches), 5)
                                                                 
print("ARI_scVIC: ")
print(ARI_scVIC)
np.save("saved/ARI_scVIC_balanced_involving_batch_effects_dropout1.npy", ARI_scVIC)

print("NMI_scVIC: ")
print(NMI_scVIC)
np.save("saved/NMI_scVIC_balanced_involving_batch_effects_dropout1.npy", NMI_scVIC)

print("ACC_scVIC: ")
print(ACC_scVIC)
np.save("saved/ACC_scVIC_balanced_involving_batch_effects_dropout1.npy", ACC_scVIC)

print("BatchMixing_scVIC: ")
print(BatchMixing_scVIC)
np.save("saved/BatchMixing_scVIC_balanced_involving_batch_effects_dropout1.npy", BatchMixing_scVIC)

print("ARI_scVIC_Louvain: ")
print(ARI_scVIC_Louvain)
np.save("saved/ARI_scVIC_Louvain_balanced_involving_batch_effects_dropout1.npy", ARI_scVIC_Louvain)

print("NMI_scVIC_Louvain: ")
print(NMI_scVIC_Louvain)
np.save("saved/NMI_scVIC_Louvain_balanced_involving_batch_effects_dropout1.npy", NMI_scVIC_Louvain)

print("ACC_scVIC_Louvain: ")
print(ACC_scVIC_Louvain)
np.save("saved/ACC_scVIC_Louvain_balanced_involving_batch_effects_dropout1.npy", ACC_scVIC_Louvain)      

print("BatchMixing_scVIC_Louvain: ")
print(BatchMixing_scVIC_Louvain)
np.save("saved/BatchMixing_scVIC_Louvain_balanced_involving_batch_effects_dropout1.npy", BatchMixing_scVIC_Louvain)
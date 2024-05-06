from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scvae.cli import train
from scvae.predict_labels import labels_predicted
import os
import pandas as pd
import numpy as np
import scanpy as sc
import scipy as sp
import os
import tensorflow as tf
import random

seed = 1


def cluster_acc(y_true, y_pred):
    
    y_true = np.array(y_true) if type(y_true) != np.ndarray else y_true
    y_pred = np.array(y_pred) if type(y_pred) != np.ndarray else y_pred
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(*ind)]) * 1.0 / y_pred.size

def normalize(adata, copy=True, highly_genes = None, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error
    if adata.X.size < 50e6: # check if adata.X is integer only if array is small
        if sp.sparse.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
    if logtrans_input:
        sc.pp.log1p(adata)
    if highly_genes != None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = highly_genes, subset=True)
    if normalize_input:
        sc.pp.scale(adata)
    return adata

i_dropout = 1
highly_genes = 500

ARI_scVAE = np.zeros((5, 10))
NMI_scVAE = np.zeros((5, 10))
ACC_scVAE = np.zeros((5, 10))

for i_cluster in range(5):
    for i_duplicate in range(10):
        
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        tf.reset_default_graph()
        
        X = pd.read_csv(
            "/Users/Healthy/Desktop/final/R/generate simulated datasets/data/Balanced Datasets/" + str(i_cluster + 3) + "/dropout" + str(i_dropout + 1) +"/counts_duplicate" + str(i_duplicate + 1) + ".csv",
                            index_col=0).values.T
        groups = pd.read_csv(
            "/Users/Healthy/Desktop/final/R/generate simulated datasets/data/Balanced Datasets/" + str(i_cluster + 3) + "/dropout" + str(i_dropout + 1) +"/groups_duplicate" + str(i_duplicate + 1) + ".csv",
                            index_col=0).values.squeeze()
                
        cell_types = np.unique(groups).tolist()
        Y = []
        for i in range(len(groups)):
            for j in range(len(cell_types)):
                if groups[i] == cell_types[j]:
                    Y.append(j)
                    continue
        X = np.ceil(X).astype(np.int)
        count_X = X
        adata = sc.AnnData(X)
        adata.obs['Group'] = Y
        adata = normalize(adata, copy=True, highly_genes=highly_genes, size_factors=True, normalize_input=True, logtrans_input=True)
        Y = np.array(adata.obs["Group"])
        high_variable = np.array(adata.var.highly_variable.index, dtype=np.int)
        X = count_X[:, high_variable]
        if not os.path.exists("data/" + str(i_cluster + 3) + "/dropout" + str(i_dropout + 1)):
            os.makedirs("data/" + str(i_cluster + 3) + "/dropout" + str(i_dropout + 1))
        pd.DataFrame(X).to_csv("data/" + str(i_cluster + 3) + "/dropout" + str(i_dropout + 1) +"/counts_duplicate" + str(i_duplicate + 1) + ".tsv", sep='\t', header=False, index=False)
        cluster_number = int(max(Y) - min(Y) + 1)   
        
        train(
            "data/" + str(i_cluster + 3) + "/dropout" + str(i_dropout + 1) +"/counts_duplicate" + str(i_duplicate + 1) + ".tsv", 
            splitting_fraction=1.0, 
            model_type="GMVAE", 
            latent_size=10,
            hidden_sizes=[128],
            latent_distribution="gaussian mixture",
            number_of_classes=cluster_number,
            batch_correction=False,
            reconstruction_distribution="zero-inflated negative binomial",
            data_directory="calculate_data/balanced/" + str(i_cluster + 3) + "/dropout" + str(i_dropout + 1),
            models_directory="models/balanced/" + str(i_cluster + 3) + "/dropout" + str(i_dropout + 1)
        )
        Y_pred = labels_predicted(
            "data/" + str(i_cluster + 3) + "/dropout" + str(i_dropout + 1) +"/counts_duplicate" + str(i_duplicate + 1) + ".tsv", 
            splitting_fraction=1.0, 
            model_type="GMVAE", 
            latent_size=10,
            hidden_sizes=[128],
            latent_distribution="gaussian mixture",
            number_of_classes=cluster_number,
            batch_correction=False,
            reconstruction_distribution="zero-inflated negative binomial",
            data_directory="calculate_data/balanced/" + str(i_cluster + 3) + "/dropout" + str(i_dropout + 1),
            models_directory="models/balanced/" + str(i_cluster + 3) + "/dropout" + str(i_dropout + 1),
            evaluation_set_kind="full"
        )
            
        accuracy = np.around(cluster_acc(Y, Y_pred), 5)
        ARI = np.around(adjusted_rand_score(Y, Y_pred), 5)
        NMI = np.around(normalized_mutual_info_score(Y, Y_pred, average_method='arithmetic'), 5)
    
        ARI_scVAE[i_duplicate] = ARI        
        NMI_scVAE[i_duplicate] = NMI
        ACC_scVAE[i_duplicate] = accuracy
        
print("ARI_scVAE: ")
print(ARI_scVAE)
np.save("saved/ARI_scVAE_balanced_dropout2.npy", ARI_scVAE)

print("NMI_scVAE: ")
print(NMI_scVAE)
np.save("saved/NMI_scVAE_balanced_dropout2.npy", NMI_scVAE)

print("ACC_scVAE: ")
print(ACC_scVAE)
np.save("saved/ACC_scVAE_balanced_dropout2.npy", ACC_scVAE)
import os              
os.environ['PYTHONHASHSEED'] = '0'
import desc          
import pandas as pd                                                    
import numpy as np                                                     
import scanpy as sc                                                                                                                                     
import sys
import anndata
from scvic.dataset import ExpressionDataset
import tensorflow as tf
import random

seed = 1
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scvic.utils import cluster_acc, entropy_batch_mixing

i_dropout = 1

ARI_DESC = np.zeros((5, 10))
NMI_DESC = np.zeros((5, 10))
ACC_DESC = np.zeros((5, 10))
BatchMixing_DESC = np.zeros((5, 10))
save_dir="result_balanced_dropout2"

for i_cluster in range(5):
    for i_duplicate in range(10):
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

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
        n_labels = len(cell_types)
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
                
        adata = anndata.AnnData(X=gene_dataset.X)
        labels_=pd.Series(labels, index=adata.obs.index, dtype='category')
        labels_.cat.categories=list(range(len(labels_.unique())))
        batches_=pd.Series(batch_indices, index=adata.obs.index, dtype='category')
        batches_.cat.categories=list(range(len(batches_.unique())))
        
        adata.obs["labels"]=labels_
        adata.obs["batches"]=batches_
        adata=desc.scale_bygroup(adata, groupby="batches")# if the the dataset has two or more batches you can use `adata=desc.scale(adata,groupby="BatchID")`
        adata=desc.train(adata,
                dims=[adata.shape[1],64,32],
                tol=0.005,
                n_clusters=n_labels,
                n_neighbors=10,
                batch_size=256,
                save_dir=str(save_dir),
                do_tsne=False,
                learning_rate=200, # the parameter of tsne
                use_GPU=False,
                num_Cores=1, #for reproducible, only use 1 cpu
                num_Cores_tsne=4,
                save_encoder_weights=False,
                save_encoder_step=3,# save_encoder_weights is False, this parameter is not used
                use_ae_weights=False,
                do_umap=False) #if do_uamp is False, it will don't compute umap coordiate
        ARI_DESC[i_cluster, i_duplicate] = np.around(adjusted_rand_score(gene_dataset.labels.squeeze(), adata.obs["desc_1.0"]), 5)
        NMI_DESC[i_cluster, i_duplicate] = np.around(normalized_mutual_info_score(gene_dataset.labels.squeeze(), adata.obs["desc_1.0"], average_method='arithmetic'), 5)
        ACC_DESC[i_cluster, i_duplicate] = np.around(cluster_acc(gene_dataset.labels.squeeze(), adata.obs["desc_1.0"]), 5)
        BatchMixing_DESC[i_cluster, i_duplicate] = np.around(entropy_batch_mixing(adata.obsm["X_Embeded_z1.0"], gene_dataset.batch_indices, gene_dataset.n_batches), 5)
print("ARI_DESC: ")
print(ARI_DESC)
np.save("saved/ARI_DESC_balanced_dropout2.npy", ARI_DESC)

print("NMI_DESC: ")
print(NMI_DESC)
np.save("saved/NMI_DESC_balanced_dropout2.npy", NMI_DESC)

print("ACC_DESC: ")
print(ACC_DESC)
np.save("saved/ACC_DESC_balanced_dropout2.npy", ACC_DESC)

print("BatchMixing_DESC: ")
print(BatchMixing_DESC)
np.save("saved/BatchMixing_DESC_balanced_dropout2.npy", BatchMixing_DESC) 

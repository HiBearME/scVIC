from preprocess import *
from network import *
from utils import *
import argparse
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import random
import numpy as np

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

seed = 1
dataname = "simulation"
distribution = "ZINB"
self_training = True
highly_genes = 500
alpha = 0.001
gamma = 0.001
learning_rate = 0.0001
batch_size = 256
update_epoch = 10
pretrain_epoch = 1000
funetrain_epoch = 2000
t_alpha = 1.0
noise_sd = 1.5
error = 0.001
gpu_option = "0"

i_dropout = 2

ARI_scziDesk = np.zeros((5,10))
ARI_scziDeskKmeans = np.zeros((5,10))
NMI_scziDesk = np.zeros((5,10))
NMI_scziDeskKmeans = np.zeros((5,10))
ACC_scziDesk = np.zeros((5,10))
ACC_scziDeskKmeans = np.zeros((5,10))

for i_ratio in range(5):
    for i_duplicate in range(10):
        
        X = pd.read_csv(
            "/Users/Healthy/Desktop/final/R/generate simulated datasets/data/Imbalanced Datasets/ratio" + str(i_ratio + 1) + "/dropout" + str(i_dropout + 1) +"/counts_duplicate" + str(i_duplicate + 1) + ".csv",
                            index_col=0).values.T
        groups = pd.read_csv(
            "/Users/Healthy/Desktop/final/R/generate simulated datasets/data/Imbalanced Datasets/ratio" + str(i_ratio + 1) + "/dropout" + str(i_dropout + 1) +"/groups_duplicate" + str(i_duplicate + 1) + ".csv",
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
        X = adata.X.astype(np.float32)
        Y = np.array(adata.obs["Group"])
        high_variable = np.array(adata.var.highly_variable.index, dtype=np.int)
        count_X = count_X[:, high_variable]
        dims = [count_X.shape[1], 256, 64, 32]
        size_factor = np.array(adata.obs.size_factors).reshape(-1, 1).astype(np.float32)
        cluster_number = int(max(Y) - min(Y) + 1)

        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        tf.reset_default_graph()
        chencluster = autoencoder(dataname, distribution, self_training, dims, cluster_number, t_alpha,
                                  alpha, gamma, learning_rate, noise_sd)
        chencluster.pretrain(X, count_X, size_factor, batch_size, pretrain_epoch, gpu_option)
        chencluster.funetrain(X, count_X, size_factor, batch_size, funetrain_epoch, update_epoch, error)
        kmeans_accuracy = np.around(cluster_acc(Y, chencluster.kmeans_pred), 5)
        kmeans_ARI = np.around(adjusted_rand_score(Y, chencluster.kmeans_pred), 5)
        kmeans_NMI = np.around(normalized_mutual_info_score(Y, chencluster.kmeans_pred, average_method='arithmetic'), 5)
        accuracy = np.around(cluster_acc(Y, chencluster.Y_pred), 5)
        ARI = np.around(adjusted_rand_score(Y, chencluster.Y_pred), 5)
        NMI = np.around(normalized_mutual_info_score(Y, chencluster.Y_pred, average_method='arithmetic'), 5)

        ARI_scziDesk[i_ratio, i_duplicate] = ARI        
        NMI_scziDesk[i_ratio, i_duplicate] = NMI
        ACC_scziDesk[i_ratio, i_duplicate] = accuracy
        
        ARI_scziDeskKmeans[i_ratio, i_duplicate] = kmeans_ARI
        NMI_scziDeskKmeans[i_ratio, i_duplicate] = kmeans_NMI
        ACC_scziDeskKmeans[i_ratio, i_duplicate] = kmeans_accuracy
        
print("ARI_scziDesk: ")
print(ARI_scziDesk)
np.save("saved/ARI_scziDesk_imbalanced_dropout3.npy", ARI_scziDesk)

print("NMI_scziDesk: ")
print(NMI_scziDesk)
np.save("saved/NMI_scziDesk_imbalanced_dropout3.npy", NMI_scziDesk)

print("ACC_scziDesk: ")
print(ACC_scziDesk)
np.save("saved/ACC_scziDesk_imbalanced_dropout3.npy", ACC_scziDesk)
                                                         
print("ARI_scziDeskKmeans: ")
print(ARI_scziDeskKmeans)
np.save("saved/ARI_scziDeskKmeans_imbalanced_dropout3.npy", ARI_scziDeskKmeans)

print("NMI_scziDeskKmeans: ")
print(NMI_scziDeskKmeans)
np.save("saved/NMI_scziDeskKmeans_imbalanced_dropout3.npy", NMI_scziDeskKmeans)

print("ACC_scziDeskKmeans: ")
print(ACC_scziDeskKmeans)
np.save("saved/ACC_scziDeskKmeans_imbalanced_dropout3.npy", ACC_scziDeskKmeans)
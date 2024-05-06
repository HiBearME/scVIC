# scVIC - Deep generative modeling of heterogeneity for scRNA-seq data

scVIC is a package for generative modeling heterogeneity for scRNA-seq data, implemented by Pytorch.

![Overview of scVIC](https://raw.githubusercontent.com/HiBearME/scVIC/master/figures/Overview.png "Overview of scVIC")
## Installation

1. Install Python 3.7.

2. Install `PyTorch <https://pytorch.org>`_. If you have an NVIDIA GPU, be sure
   to install a version of PyTorch that supports it. scVIC runs much faster
   with GPUs.

3. Install scVIC from GitHub:

```bash
git clone git://github.com/HiBearME/scVIC.git
cd scVIC
python setup.py install --user
```


## How to use scVIC
1. Load Data

Our datasets loading and preprocessing module is based on scVI.
For datasets available in scVI, users can directly obtain it through Python code.
For example, load the online dataset ``RETINA`` available in scVI as

```python
from scvi.dataset import RetinaDataset
from scvic.dataset import ExpressionDataset
save_path = "../Datasets/Biological Datasets"
gene_dataset = RetinaDataset(save_path=save_path)
gene_dataset.filter_genes_by_count(per_batch=True)
gene_dataset.make_gene_names_lower()
gene_dataset.subsample_genes(4000)
retina_dataset = ExpressionDataset()
retina_dataset.load_dataset_from_scVI(gene_dataset)
```

For datasets more generic as AnnData, users can transform it into data frame in scVI.
For example, load the local dataset ``BACH`` in frame of AnnData as

```python
from scvi.dataset import AnnDatasetFromAnnData
from scvic.dataset import ExpressionDataset
import scanpy as sc
adata = sc.AnnData(X) # X is the expression matrix of the BACH dataset.
adata.obs['cell_types'] = cell_types # cell_types is the types vector of the BACH dataset.
gene_dataset = AnnDatasetFromAnnData(adata)
gene_dataset.subsample_genes(1000)
gene_dataset.make_gene_names_lower()
bach_dataset = ExpressionDataset()
bach_dataset.load_dataset_from_scVI(gene_dataset)
```

2. Train

For example, train RETINA dataset as (the meaning of parameters showed as comments)

```python
n_epochs = 400 # given number of epochs (Default as scVI)
lr = 0.001 # given learning rate (Default as scVI)
use_cuda = True # whether use gpu or not
use_batches = True # whether remove batch effects or not

# Initialize scVIC model by CVAE class using dateset info: the number of genes, cell types, and batches.
# If without batch effect removal, please set use_batches as False.
# n_latent denotes the dimensionality of latent variable (Default as scVI)
retina_cvae = CVAE(retina_dataset.nb_genes, n_labels=retina_dataset.n_labels, n_batch=retina_dataset.n_batches * use_batches, n_latent=10)

# Initialize train setup of scVIC by CTrainer class using initialized scVIC model and RNA-seq datasets.
# train_size is the proportion of train dataset. (Default as using all samples)
# n_epochs_pre_train and n_epochs_kl_warmup is the computational trick of neural networks. (Half of epoches as default)
ctrainer = CTrainer(
    retina_cvae,
    retina_dataset,
    train_size=1.0,
    use_cuda=use_cuda,
    n_epochs_kl_warmup=200,
    n_epochs_pre_train=200)

# Actual train by train function of CTrainer class.
ctrainer.train(n_epochs=n_epochs, lr=lr)
```
3. Get Embedding and labels predicted (the meaning of parameters showed as comments)

```python
import numpy as np

# Inference of scVIC by Posterior class, using create_posterior function of CTrainer class given scVIC model and dataset.
full = ctrainer.create_posterior(ctrainer.model, retina_dataset, indices=np.arange(len(retina_dataset)))
# reset inference batch size (Optional, not reset is total fine)
full = full.update({"batch_size":32})
# Complete inference by get_latent function of  Posterior class.
latent, labels_pred = full.sequential().get_latent()
```

## Reproducibility

To reproduce the results of original manuscript, please check in `notebooks/repoducibility`.


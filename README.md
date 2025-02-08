# scVIC - Deep Generative Modeling of Heterogeneity for scRNA-seq Data

**scVIC** is a package for generative modeling heterogeneity for scRNA-seq data, implemented in **PyTorch**.

![Overview of scVIC](https://raw.githubusercontent.com/HiBearME/scVIC/master/figures/Overview.png "Overview of scVIC")
## Installation

1. Install **Conda** and create a virtual environment with `python==3.7`:

   ```bash
   conda create -n scVIC python==3.7
   conda activate scVIC
   ```

2. Install [PyTorch](https://pytorch.org) in the virtual environment. If you have an **NVIDIA GPU**, make sure to install a version of PyTorch that supports it. PyTorch performs much faster with an NVIDIA GPU. For maximum compatibility, we currently recommend installing `pytorch==1.12.1`.

3. Install scVIC from GitHub:

   ```bash
   git clone git://github.com/HiBearME/scVIC.git
   cd scVIC
   pip install .
   ```


## Usage
1. Load Data

scVIC utilizes **scVI** for dataset loading and preprocessing. If your dataset is available in scVI, you can directly retrieve it.  

For example, to load the **RETINA** dataset from scVI:

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

For **AnnData**-formatted datasets, you can convert them to a scVI-compatible format.  

For example, to load the **BACH** dataset in format of **AnnData**:

```python
from scvi.dataset import AnnDatasetFromAnnData
from scvic.dataset import ExpressionDataset
import scanpy as sc
adata = sc.AnnData(X) # X: Expression matrix of the BACH dataset.
adata.obs['cell_types'] = cell_types # cell_types: Cell type labels.
gene_dataset = AnnDatasetFromAnnData(adata)
gene_dataset.subsample_genes(1000)
gene_dataset.make_gene_names_lower()
bach_dataset = ExpressionDataset()
bach_dataset.load_dataset_from_scVI(gene_dataset)
```

2. Train

For example, train scVIC on **RETINA** dataset:

```python
from scvic.models import CVAE
from scvic.inference import CTrainer

# Define training parameters
n_epochs = 400 # Number of training epochs (default: 400, same as scVI)
lr = 0.001 # Learning rate (default: 0.001, same as scVI)
use_cuda = True # Enable GPU acceleration
use_batches = True # Take batch effects into account (set to False to disable)

# Initialize the scVIC model using dataset information: the number of genes, cell types, and batches.
# If batch effect removal is not needed, set use_batches to False.
# n_latent denotes the dimensionality of the latent variable (default: 10, same as scVI)
retina_cvae = CVAE(
   retina_dataset.nb_genes, 
   n_labels=retina_dataset.n_labels,
   n_batch=retina_dataset.n_batches * use_batches, 
   n_latent=10)

# Set up the training process using CTrainer with the initialized scVIC model and RNA-seq dataset.
# train_size defines the proportion of the dataset used for training (default: 100%)
# n_epochs_pre_train and n_epochs_kl_warmup help stabilize training (default: half of total epochs)
ctrainer = CTrainer(
    retina_cvae,
    retina_dataset,
    train_size=1.0,
    use_cuda=use_cuda,
    n_epochs_kl_warmup=200,
    n_epochs_pre_train=200)

# Train the model
ctrainer.train(n_epochs=n_epochs, lr=lr)
```
3. Get Embedding and Predicted Labels

```python
import numpy as np

# Create posterior inference object
full = ctrainer.create_posterior(ctrainer.model, retina_dataset, indices=np.arange(len(retina_dataset)))
# Optionally, update batch size for inference
full = full.update({"batch_size":32})
# Perform inference
latent, labels_pred = full.sequential().get_latent()
```

## Reproducibility

To reproduce the results of original manuscript, please check in `notebooks/repoducibility`.


## Variant

If you wish to use scVIC with Python 3.9, please switch to the **python3.9** branch. However, since the Python version and dependency environment have changed in this branch, reproducibility is not guaranteed.

## Citation

Jiankang Xiong, Fuzhou Gong, Liang Ma and Lin Wan. "scVIC: deep generative modeling of heterogeneity for scRNA-seq data." Bioinformatics Advances (2024): vbae086, https://doi.org/10.1093/bioadv/vbae086 .
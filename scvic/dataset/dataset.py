import torch
from scvi.dataset import GeneExpressionDataset
import numpy as np
import pandas as pd
import scanpy as sc
from typing import Optional, Union, Dict, Callable, List, Tuple
import scipy.sparse as sp_sparse
from functools import partial
import logging

logger = logging.getLogger(__name__)


class ExpressionDataset(GeneExpressionDataset):

    def __init__(self, new_n_genes=None, batch_correction=False):
        super().__init__()
        self.new_n_genes = new_n_genes
        self.batch_correction = batch_correction
        self._X_dropout_removal = None
        self.cell_measurements_col_mappings = dict()

    def load_dataset_from_scVI(self, dataset: GeneExpressionDataset):
        # registers
        self.dataset_versions = dataset.dataset_versions
        self.gene_attribute_names = dataset.gene_attribute_names
        self.cell_attribute_names = dataset.cell_attribute_names
        self.cell_categorical_attribute_names = dataset.cell_categorical_attribute_names
        self.attribute_mappings = dataset.attribute_mappings
        self.cell_measurements_col_mappings = dataset.cell_measurements_col_mappings

        # initialize attributes
        self._X = dataset.X
        self._batch_indices = dataset.batch_indices
        self._labels = dataset.labels
        self.n_batches = dataset.n_batches
        self.n_labels = dataset.n_labels
        self.gene_names = dataset.gene_names
        self.cell_types = dataset.cell_types
        self.local_means = dataset.local_means
        self.local_vars = dataset.local_vars
        self._norm_X = dataset.norm_X
        self._corrupted_X = dataset.corrupted_X

        # attributes that should not be set by initialization methods
        self.protected_attributes = dataset.protected_attributes

        for attr_name in self.cell_attribute_names:
            if not hasattr(self, attr_name):
                setattr(self, attr_name, getattr(dataset, attr_name))

        for attr_name in self.gene_attribute_names:
            if not hasattr(self, attr_name):
                setattr(self, attr_name, getattr(dataset, attr_name))

    @property
    def X_dropout_removal(self) -> Union[sp_sparse.csr_matrix, np.ndarray]:
        """Returns the corrupted version of X."""
        return self._X_dropout_removal

    @X_dropout_removal.setter
    def X_dropout_removal(self, X_dropout_removal: Union[sp_sparse.csr_matrix, np.ndarray]):
        self._X_dropout_removal = X_dropout_removal
        self.register_dataset_version("X_dropout_removal")

    def keep_highly_variable_genes_by_seurat(
            self,
            new_n_genes: Optional[int] = None,
            batch_correction: Optional[bool] = None
    ):
        if new_n_genes is None:
            new_n_genes = self.new_n_genes
        if batch_correction is None:
            batch_correction = self.batch_correction
        self.filter_genes_by_count(min_count=1)
        self.filter_cells_by_count(min_count=1)
        obs = pd.DataFrame(
            data=dict(batch=self.batch_indices.squeeze()),
            index=np.arange(self.nb_cells).astype(str)
        ).astype("category")

        counts = self.X.copy()
        adata = sc.AnnData(X=counts, obs=obs)
        batch_key = "batch" if (batch_correction and self.n_batches >= 2) else None
        sc.pp.normalize_per_cell(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(
            adata=adata,
            min_mean=0.0125,
            max_mean=3,
            min_disp=0.5,
            n_top_genes=new_n_genes,
            batch_key=batch_key,
            subset=True,
            flavor="seurat_v3"
         )
        genes_infos = adata.var
        subset_genes = np.array(genes_infos["highly_variable"].index, dtype=np.int)
        self.update_genes(subset_genes)

    def collate_fn_builder(
            self,
            add_attributes_and_types: Dict[str, type] = None,
            override: bool = False,
            corrupted=False,
            dropout_removal=False,
    ) -> Callable[[Union[List[int], np.ndarray]], Tuple[torch.Tensor, ...]]:
        """Returns a collate_fn with the requested shape/attributes"""

        if override:
            attributes_and_types = dict()
        else:
            if dropout_removal:
                X_chosen = ("X_dropout_removal", np.float32)
            elif corrupted:
                X_chosen = ("corrupted_X", np.float32)
            else:
                X_chosen = ("X", np.float32)
            attributes_and_types = dict(
                [
                    X_chosen,
                    ("local_means", np.float32),
                    ("local_vars", np.float32),
                    ("batch_indices", np.int64),
                    ("labels", np.int64),
                ]
            )

        if add_attributes_and_types is None:
            add_attributes_and_types = dict()
        attributes_and_types.update(add_attributes_and_types)
        return partial(self.collate_fn_base, attributes_and_types)

    def make_gene_names_lower(self):
        logger.info("Making gene names lower case")
        self.gene_names = np.char.lower(self.gene_names)
import os

import anndata
from scvi.inference import Posterior, UnsupervisedTrainer
from scvi.inference.trainer import SequentialSubsetSampler
from scvi.models.utils import one_hot
from scvi.inference.posterior import plot_imputation
import numpy as np
import torch
from typing import List, Optional, Union, Tuple
import logging
import time
import pandas as pd
import scanpy as sc

from torch.distributions import Normal

from scvic.utils import entropy_batch_mixing
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


class CPosterior(Posterior):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def get_latent(self, give_mean: Optional[bool] = True) -> Tuple:
        """Output posterior z mean or sample, batch index, and label

        :param sample: z mean or z sample
        :return: three np.ndarrays, latent, batch_indices, labels
        """
        latent = []
        labels_pred = []
        for tensors in self:
            sample_batch, local_l_mean, local_l_var, batch_index, label = tensors
            z, qc = self.model.sample_from_posterior_z(
                sample_batch, batch_index, give_mean=give_mean
            )
            latent += [z.cpu()]
            labels_pred += [qc.argmax(dim=1).cpu()]
        return (
            np.array(torch.cat(latent)),
            np.array(torch.cat(labels_pred), dtype=int)
        )

    @torch.no_grad()
    def elbo(self) -> torch.Tensor:
        """Returns the Evidence Lower Bound associated to the object.

        """
        elbo = self.compute_elbo()
        logger.debug("ELBO : %.4f" % elbo)
        return elbo

    elbo.mode = "min"

    @torch.no_grad()
    def compute_elbo(self):
        """ Computes the ELBO.

        The ELBO is the reconstruction error + the KL divergences
        between the variational distributions and the priors.
        It differs from the marginal log likelihood.
        Specifically, it is a lower bound on the marginal log likelihood
        plus a term that is constant with respect to the variational distribution.
        It still gives good insights on the modeling of the data, and is fast to compute.
        """
        # Iterate once over the posterior and compute the elbo
        elbo = 0
        for tensors in self.sequential():
            sample_batch, local_l_mean, local_l_var, batch_index, labels = tensors
            # kl_divergence_global (scalar) should be common across all batches after training
            # reconst_loss, kl_divergence, kl_divergence_global, _ = self.model(
            reconst_loss, kl_divergence, kl_divergence_global = self.model(
                sample_batch,
                local_l_mean,
                local_l_var,
                batch_index=batch_index
            )
            elbo += torch.sum(reconst_loss + kl_divergence).item()
        n_samples = len(self.indices)
        elbo += kl_divergence_global
        return elbo / n_samples

    @torch.no_grad()
    def entropy_batch_mixing(self, **kwargs):
        """Returns the object's entropy batch mixing.

        """
        if self.gene_dataset.n_batches > 1:
            latent, _ = self.sequential().get_latent()
            batches = self.gene_dataset.batch_indices
            be_score = entropy_batch_mixing(latent, batches, self.gene_dataset.n_batches, **kwargs)
            logger.debug("Entropy batch mixing : {}".format(be_score))
            return be_score

    @torch.no_grad()
    def imputation(
            self,
            n_samples: Optional[int] = 1,
            transform_batch: Optional[Union[int, List[int]]] = None,
    ) -> np.ndarray:
        """Imputes px_rate over self cells

        :param n_samples:
        :param transform_batch: Batches to condition on.
          If transform_batch is:
            - None, then real observed batch is used
            - int, then batch transform_batch is used
            - list of int, then px_rates are averaged over provided batches.
        :return: (n_samples, n_cells, n_genes) px_rates squeezed array
        """
        if (transform_batch is None) or (isinstance(transform_batch, int)):
            transform_batch = [transform_batch]
        imputed_arr = []
        for batch in transform_batch:
            imputed_list_batch = []
            for tensors in self:
                sample_batch, local_l_mean, local_l_var, batch_index, label = tensors
                px_rate = self.model.get_sample_rate(
                    sample_batch,
                    batch_index=batch_index,
                    n_samples=n_samples,
                    transform_batch=batch,
                )
                imputed_list_batch += [np.array(px_rate.cpu())]
            imputed_arr.append(np.concatenate(imputed_list_batch))
        imputed_arr = np.array(imputed_arr)
        # shape: (len(transformed_batch), n_samples, n_cells, n_genes) if n_samples > 1
        # else shape: (len(transformed_batch), n_cells, n_genes)
        return imputed_arr.mean(0).squeeze()

    @torch.no_grad()
    def imputation_benchmark(self, n_samples: int = 8,
                             show_plot: bool = True,
                             title_plot: str = "imputation",
                             save_path: str = ""):
        return super().imputation_benchmark(n_samples, show_plot, title_plot, save_path)

    @torch.no_grad()
    def imputation_list(self, n_samples: int = 1, situation="corrupted") -> tuple:
        """Imputes data's gene counts from corrupted data.

        :return: Original gene counts and imputations after corruption.
        """
        original_list = []
        imputed_list = []
        batch_size = 10000  # self.data_loader_kwargs["batch_size"] // n_samples
        if situation is "corrupted":
            for tensors, corrupted_tensors in zip(
                    self.uncorrupted().sequential(batch_size=batch_size),
                    self.corrupted().sequential(batch_size=batch_size),
            ):
                batch = tensors[0]
                actual_batch_size = batch.size(0)
                dropout_batch, _, _, batch_index, labels = corrupted_tensors
                px_rate = self.model.get_sample_rate(
                    dropout_batch, batch_index=batch_index, y=labels, n_samples=n_samples
                )

                indices_dropout = torch.nonzero(batch - dropout_batch)
                if indices_dropout.size() != torch.Size([0]):
                    i = indices_dropout[:, 0]
                    j = indices_dropout[:, 1]

                    batch = batch.unsqueeze(0).expand(
                        (n_samples, batch.size(0), batch.size(1))
                    )
                    original = np.array(batch[:, i, j].view(-1).cpu())
                    imputed = np.array(px_rate[..., i, j].view(-1).cpu())

                    cells_index = np.tile(np.array(i.cpu()), n_samples)

                    original_list += [
                        original[cells_index == i] for i in range(actual_batch_size)
                    ]
                    imputed_list += [
                        imputed[cells_index == i] for i in range(actual_batch_size)
                    ]
                else:
                    original_list = np.array([])
                    imputed_list = np.array([])
            return original_list, imputed_list
        elif situation is "dropout removal":
            for tensors_dropout_removal, tensors_dropout_not_removal in zip(
                    self.dropout_not_removal().sequential(batch_size=batch_size),
                    self.dropout_removal().sequential(batch_size=batch_size),
            ):
                batch = tensors_dropout_removal[0]
                actual_batch_size = batch.size(0)
                dropout_batch, _, _, batch_index, labels = tensors_dropout_not_removal
                px_rate = self.model.get_sample_rate(
                    dropout_batch, batch_index=batch_index, n_samples=n_samples
                )

                indices_dropout = torch.nonzero(batch - dropout_batch)
                if indices_dropout.size() != torch.Size([0]):
                    i = indices_dropout[:, 0]
                    j = indices_dropout[:, 1]

                    batch = batch.unsqueeze(0).expand(
                        (n_samples, batch.size(0), batch.size(1))
                    )
                    original = np.array(batch[:, i, j].view(-1).cpu())
                    imputed = np.array(px_rate[..., i, j].view(-1).cpu())

                    cells_index = np.tile(np.array(i.cpu()), n_samples)

                    original_list += [
                        original[cells_index == i] for i in range(actual_batch_size)
                    ]
                    imputed_list += [
                        imputed[cells_index == i] for i in range(actual_batch_size)
                    ]
                else:
                    original_list = np.array([])
                    imputed_list = np.array([])
            return original_list, imputed_list
        else:
            raise NotImplementedError()

    @torch.no_grad()
    def imputation_score(
            self, original_list: List = None, imputed_list: List = None, n_samples: int = 1,
            situation="corrupted"
    ) -> float:
        """Computes median absolute imputation error.

        """
        if original_list is None or imputed_list is None:
            original_list, imputed_list = self.imputation_list(n_samples=n_samples, situation=situation)

        original_list_concat = np.concatenate(original_list)
        imputed_list_concat = np.concatenate(imputed_list)
        are_lists_empty = (len(original_list_concat) == 0) and (
                len(imputed_list_concat) == 0
        )
        if are_lists_empty:
            logger.info(
                "No difference between corrupted dataset and uncorrupted dataset"
            )
            return 0.0
        else:
            return np.median(np.abs(original_list_concat - imputed_list_concat))

    @torch.no_grad()
    def imputation_benchmark(
            self,
            n_samples: int = 8,
            show_plot: bool = True,
            title_plot: str = "imputation",
            save_path: str = "",
            situation="corrupted"
    ) -> Tuple:
        """Visualizes the model imputation performance.

        """
        original_list, imputed_list = self.imputation_list(n_samples=n_samples, situation=situation)
        # Median of medians for all distances
        median_score = self.imputation_score(
            original_list=original_list, imputed_list=imputed_list
        )

        # Mean of medians for each cell
        imputation_cells = []
        for original, imputed in zip(original_list, imputed_list):
            has_imputation = len(original) and len(imputed)
            imputation_cells += [
                np.median(np.abs(original - imputed)) if has_imputation else 0
            ]
        mean_score = np.mean(imputation_cells)

        logger.debug(
            "\nMedian of Median: %.4f\nMean of Median for each cell: %.4f"
            % (median_score, mean_score)
        )

        plot_imputation(
            np.concatenate(original_list),
            np.concatenate(imputed_list),
            show_plot=show_plot,
            title=os.path.join(save_path, title_plot),
        )
        return original_list, imputed_list

    def dropout_removal(self) -> "Posterior":
        """Corrupts gene counts.

        """
        return self.update(
            {"collate_fn": self.gene_dataset.collate_fn_builder(dropout_removal=True)}
        )

    def dropout_not_removal(self) -> "Posterior":
        """Uncorrupts gene counts.

        """
        return self.update({"collate_fn": self.gene_dataset.collate_fn_builder()})

    @torch.no_grad()
    def standard_scale_sampler(
            self,
            n_samples: Optional[int] = 5000,
            batch_sampler: str = "prior",
            cluster_sampler: str = "posterior"
    ) -> dict:

        batchid = range(self.gene_dataset.n_batches)
        if batch_sampler is "prior":
            counts = np.zeros(self.gene_dataset.n_batches)
            for batch in batchid:
                counts[batch] = sum(self.gene_dataset.batch_indices == batch)
            p_batch = np.array(counts) / sum(counts)
        elif batch_sampler is "uniform":
            p_batch = np.ones(self.gene_dataset.n_batches) / self.gene_dataset.n_batches
        else:
            raise NotImplementedError()

        samples_for_each_batch = (n_samples * p_batch / p_batch.max()).astype(int)

        if cluster_sampler is "posterior":
            p_cluster = self.model.pi.squeeze().cpu().detach().numpy()
        elif cluster_sampler is "uniform":
            p_cluster = np.ones(self.gene_dataset.n_labels) / self.gene_dataset.n_labels
        else:
            raise NotImplementedError()

        scales_for_each_batch = []
        clusters_for_each_batch = []

        for batch_idx in batchid:
            samples_for_each_cluster = (samples_for_each_batch[batch_idx] * p_cluster).astype(int)
            cluster_index = [
                i * torch.ones(samples, device=self.model.mu.device)
                for i, samples in enumerate(samples_for_each_cluster)]
            cluster_index = torch.cat(cluster_index).view(-1, 1)
            clusters_for_each_batch.append(cluster_index)
            cluster_index_one_hot = one_hot(cluster_index, self.gene_dataset.n_labels)
            mu = torch.mm(cluster_index_one_hot, self.model.mu)
            # if self.model.covariance_type == "diag":
            #     var = torch.mm(cluster_index_one_hot, torch.exp(self.model.logvar)) + 1e-4
            # elif self.model.covariance_type == "spherical":
            #     logvar = self.model.logvar.repeat(1, self.model.n_latent)
            #     var = torch.mm(cluster_index_one_hot, torch.exp(logvar)) + 1e-4
            # else:
            var = torch.ones_like(mu)
            z = Normal(mu, torch.sqrt(var)).sample()
            batch = torch.ones(z.size(0), 1, device=self.model.mu.device) * batch_idx
            scales_for_each_batch.append(self.model.get_sample_scale_from_z(z, batch))
        return dict(scales=scales_for_each_batch, labels=clusters_for_each_batch)

    @torch.no_grad()
    def bayes_factors(
            self,
            n_samples: Optional[int] = 5000,
            batch_sampler: str = "prior",
            cluster_sampler: str = "posterior"
    ):

        eps = 1e-8
        sampler = self.standard_scale_sampler(n_samples, batch_sampler, cluster_sampler)
        scales_for_all_batch = sampler["scales"]
        labels_for_all_batch = sampler["labels"]
        bayes_factors = []
        for label in np.arange(self.gene_dataset.n_labels):
            counts = 0
            over = 0
            for batch in np.arange(self.gene_dataset.n_batches):
                scales = scales_for_all_batch[batch]
                labels = labels_for_all_batch[batch].squeeze()
                scales_for_the_label = scales[labels == label]
                indices = torch.randperm(scales_for_the_label.size(0))
                scales_not_for_the_label = scales[labels != label][indices, :]
                counts = counts + scales_for_the_label.size(0)
                over = over + (scales_for_the_label > scales_not_for_the_label).sum(dim=0).cpu().numpy()

            proba_H1 = over / float(counts)
            proba_H2 = 1.0 - proba_H1

            bayes_factors.append(np.log(proba_H1 + eps) - np.log(proba_H2 + eps))

        gene_names = self.gene_dataset.gene_names
        res = pd.DataFrame(np.array(bayes_factors).T, index=gene_names)
        res.columns = np.arange(self.gene_dataset.n_labels)
        return res

    @torch.no_grad()
    def calibration(self, transform_batch: Union[None, int, str] = "max", n_samples: int = 1):
        if transform_batch is "max":
            batchid = range(self.gene_dataset.n_batches)
            counts = np.zeros(self.gene_dataset.n_batches)
            for batch in batchid:
                counts[batch] = sum(self.gene_dataset.batch_indices == batch)
            transform_batch = counts.argmax()

        calibration = []
        for tensors in self:
            sample_batch, local_l_mean, local_l_var, batch_index, labels = tensors
            px_rate = self.model.get_sample_scale(
                sample_batch, batch_index=batch_index, n_samples=n_samples, transform_batch=transform_batch
            )
            calibration += [px_rate.cpu()]

        return np.array(torch.cat(calibration))

    def sequential(self, batch_size: Optional[int] = 1024) -> "CPosterior":
        """Returns a copy of the object that iterate over the data sequentially.

        :param batch_size: New batch size.
        """
        return self.update(
            {
                "batch_size": batch_size,
                "sampler": SequentialSubsetSampler(indices=self.indices),
                "drop_last": False
            }
        )


class CTrainer(UnsupervisedTrainer):
    def __init__(self, model, gene_dataset, train_size=None, test_size=None,
                 n_epochs_kl_warmup=None, n_epochs_pre_train=None, tol=0.001,
                 # weights=1.0,
                 tol_start=10, extra_steps=1, *args, **kwargs):
        early_stopping_kwargs = {
            "early_stopping_metric": "elbo",
            "on": "test_set",
            "threshold": 0,
            "posterior_class": CPosterior
        }
        super().__init__(model, gene_dataset,
                         n_epochs_kl_warmup=n_epochs_kl_warmup,
                         early_stopping_kwargs=early_stopping_kwargs,
                         *args,
                         **kwargs)
        if type(self) is CTrainer:
            (
                self.train_set,
                self.test_set,
                self.validation_set,
            ) = self.train_test_validation(model, gene_dataset, train_size,
                                           test_size, CPosterior)
            self.train_set.to_monitor = ["elbo"]
            self.test_set.to_monitor = ["elbo"]
            self.validation_set.to_monitor = ["elbo"]
            self.n_samples = len(self.train_set.indices)
            self.n_epochs_pre_train = 0 if n_epochs_pre_train is None else n_epochs_pre_train
            self.extra_steps = extra_steps
            self.clustering = None
            self.tol = tol
            self.tol_start = tol_start
            # self.weights = weights

    def on_training_loop(self, tensors_list):
        # self.current_loss, latent_variable = self.loss(*tensors_list)
        self.current_loss = self.loss(*tensors_list)
        self.optimizer.zero_grad()
        self.current_loss.backward()
        self.optimizer.step()
        # if latent_variable["qc"] is not None and self.weights < 1.0:
        #     z = latent_variable["z"]
        #     qc = latent_variable["qc"]
        #     self.more_steps_for_gmm(z, qc, weights=self.weights)

    def train(self, n_epochs=800, lr=1e-2, params=None,
              **extras_kwargs):
        super().train(
            n_epochs=n_epochs,
            lr=lr,
            params=params,
            **extras_kwargs
        )

    def loss(self, tensors):
        sample_batch, local_l_mean, local_l_var, batch_index, _ = tensors

        # reconst_loss, kl_divergence_local, kl_divergence_global, latent_variable = self.model(
        reconst_loss, kl_divergence_local, kl_divergence_global = self.model(
            sample_batch, local_l_mean, local_l_var, batch_index
        )
        loss = (
                self.n_samples
                * torch.mean(reconst_loss + self.kl_weight * kl_divergence_local)
                + kl_divergence_global
        )
        if self.normalize_loss:
            loss = loss / self.n_samples
        # return loss, latent_variable
        return loss

    def train_test_validation(
            self,
            model=None,
            gene_dataset=None,
            train_size=0.1,
            test_size=None,
            type_class=CPosterior):

        return super().train_test_validation(
            model=model,
            gene_dataset=gene_dataset,
            train_size=train_size,
            test_size=test_size,
            type_class=type_class
        )

    def create_posterior(
            self,
            model=None,
            gene_dataset=None,
            shuffle=False,
            indices=None,
            type_class=CPosterior,
    ):
        return super().create_posterior(
            model=model, gene_dataset=gene_dataset,
            shuffle=shuffle, indices=indices,
            type_class=type_class)

    @torch.no_grad()
    def compute_running_parameters(self, z=None, qc=None, tol=0.001, tol_start=None,
                                   # weights=1.0
                                   ):
        continue_training = True
        tol_start = self.tol_start if tol_start is None else tol_start
        if z is None:
            z_ = []
            qc_ = []
            for tensors in self.train_set.sequential():
                sample_batch, _, _, batch_index, _ = tensors

                if self.model.log_variational:
                    sample_batch = torch.log(1 + sample_batch)

                _, _, z = self.model.z_encoder(sample_batch, batch_index)
                qc = self.model.classifier(z)

                z_.append(z)
                qc_.append(qc)

            z = torch.cat(z_)
            qc = torch.cat(qc_)
        # if not self.benchmark and tol >= 0 and tol_start <= self.epoch + 2 and weights == 1.0:
        if not self.benchmark and tol >= 0 and tol_start <= self.epoch + 2:
            clustering = qc.argmax(dim=-1)
            if self.clustering is not None:
                delta_clustering = sum(clustering != self.clustering).item() / len(clustering)
                if delta_clustering < tol:
                    logger.info(
                        "Tolerance achieved : "
                        "ratio of clustering labels changed under "
                        + str(tol)
                    )
                    continue_training = False
                else:
                    self.clustering = clustering
            else:
                self.clustering = clustering

        if continue_training:
            qc_sum = qc.sum(dim=0, keepdim=True)
            # mu = qc.t().mm(z) / qc_sum.t()
            # self.model.mu = (1 - weights) * self.model.mu + weights * mu
            self.model.mu = qc.t().mm(z) / qc_sum.t()
            # if not self.model.fix_mixture:
            #     pi = qc_sum / qc.shape[0]
            #     self.model.pi = (1 - weights) * self.model.pi + weights * pi
            self.model.pi = qc_sum / qc.shape[0]

            # if self.model.covariance_type == "diag":
            #     avg_X2 = qc.t().mm(z * z) / qc_sum.t()
            #     avg_means2 = self.model.mu ** 2
            #     avg_X_means = self.model.mu * qc.t().mm(z) / qc_sum.t()
            #     self.model.logvar = torch.log(
            #         avg_X2 - 2 * avg_X_means + avg_means2 + 1e-8)
            #     # self.model.logvar = (1 - weights) * self.model.logvar + weights * torch.log(
            #     #     avg_X2 - 2 * avg_X_means + avg_means2 + 1e-8)
            #
            # elif self.model.covariance_type == "spherical":
            #     avg_X2 = qc.t().mm(z * z) / qc_sum.t()
            #     avg_means2 = self.model.mu ** 2
            #     avg_X_means = self.model.mu * qc.t().mm(z) / qc_sum.t()
            #     self.model.logvar = torch.log(
            #         (avg_X2 - 2 * avg_X_means + avg_means2).mean(dim=-1, keepdim=True) + 1e-8)
            #     # self.model.logvar = (1 - weights) * self.model.logvar + weights * torch.log(
            #     #     (avg_X2 - 2 * avg_X_means + avg_means2).mean(dim=-1, keepdim=True) + 1e-8)

        return z, qc, continue_training

    @torch.no_grad()
    def more_steps_for_gmm(self, z=None, qc=None, max_steps=1, tol=0.001,
                           # weights=1.0
                           ):
        continue_training = True
        for i in range(max_steps):
            z, qc, continue_training = self.compute_running_parameters(
                z, qc, tol,
                # weights=weights
            )
            if not continue_training:
                break
        return continue_training

    @torch.no_grad()
    def on_epoch_begin(self):
        if self.n_epochs_pre_train == self.epoch:
            z_ = []
            for tensors in self.train_set.sequential():
                sample_batch, _, _, batch_index, _ = tensors

                if self.model.log_variational:
                    sample_batch = torch.log(1 + sample_batch)

                _, _, z = self.model.z_encoder(sample_batch, batch_index)
                z_.append(z)
            z = torch.cat(z_)

            if not self.model.n_labels:
                data = anndata.AnnData(X=z.cpu().detach().numpy())
                sc.pp.neighbors(data)
                sc.tl.louvain(data, resolution=self.model.resolution)
                self.model.n_labels = len(np.unique(data.obs['louvain']))

            GMM = GaussianMixture(n_components=self.model.n_labels).fit(
                z.cpu().detach().numpy())
            mean_ = torch.from_numpy(GMM.means_).to(torch.float32)
            weights_ = torch.from_numpy(GMM.weights_).to(torch.float32)
            self.model.mu = mean_.cuda() if self.use_cuda else mean_
            self.model.pi = weights_.cuda() if self.use_cuda else weights_
            self.model.pretrain = False

    @torch.no_grad()
    def on_epoch_end(self):
        continue_training = True
        # if self.n_epochs_pre_train <= self.epoch and self.weights == 1.0:
        if self.n_epochs_pre_train <= self.epoch:
            continue_training = self.more_steps_for_gmm(
                max_steps=self.extra_steps, tol=self.tol)
        if not continue_training:
            return continue_training
        self.compute_metrics()
        on = self.early_stopping.on
        early_stopping_metric = self.early_stopping.early_stopping_metric
        save_best_state_metric = self.early_stopping.save_best_state_metric
        if self.frequency is not None and save_best_state_metric is not None and on is not None:
            if self.early_stopping.update_state(
                    self.history[save_best_state_metric + "_" + on][-1]
            ):
                self.best_state_dict = self.model.state_dict()
                self.best_epoch = self.epoch

        if self.frequency is not None and early_stopping_metric is not None and on is not None:
            continue_training, reduce_lr = self.early_stopping.update(
                self.history[early_stopping_metric + "_" + on][-1]
            )
            if reduce_lr:
                logger.info("Reducing LR on epoch {}.".format(self.epoch))
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] *= self.early_stopping.lr_factor

        return continue_training

    @torch.no_grad()
    def compute_metrics(self):
        begin = time.time()
        epoch = self.epoch + 1
        if self.frequency and (
                epoch == 0 or epoch == self.n_epochs or (epoch % self.frequency == 0)
        ):
            with torch.set_grad_enabled(False):
                self.model.eval()
                logger.debug("\nEPOCH [%d/%d]: " % (epoch, self.n_epochs))

                for name, posterior in self._posteriors.items():
                    message = " ".join([s.capitalize() for s in name.split("_")[-2:]])
                    if posterior.nb_cells < 5:
                        logging.debug(
                            message + " is too small to track metrics (<5 samples)"
                        )
                        continue
                    if hasattr(posterior, "to_monitor"):
                        for metric in posterior.to_monitor:
                            if metric not in self.metrics_to_monitor:
                                logger.debug(message)
                                result = getattr(posterior, metric)()
                                self.history[metric + "_" + name] += [result]
                    for metric in self.metrics_to_monitor:
                        result = getattr(posterior, metric)()
                        self.history[metric + "_" + name] += [result]
                self.model.train()
        self.compute_metrics_time += time.time() - begin

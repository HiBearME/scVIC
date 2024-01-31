import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence as kl
import logging
from scvic.utils import log_zinb_positive, log_nb_positive
from torch.distributions import Categorical

from scvic.models.modules import EncoderList, DecoderList
from scvi.models.utils import one_hot, broadcast_labels

from typing import Tuple, Dict, Union, List, Optional

torch.backends.cudnn.benchmark = True
logger = logging.getLogger(__name__)


class CVAE(nn.Module):

    def __init__(
            self,
            n_input: int,
            n_labels: Optional[int] = None,
            resolution: Union[float, int, None] = 1.0,
            n_hidden: Union[int, List[int]] = 128,
            n_layers: Optional[int] = 1,
            n_batch: int = 0,  # 默认没有batch
            n_latent: int = 10,
            dropout_rate: float = 0.1,
            dispersion: str = "gene",
            log_variational: bool = True,
            reconstruction_loss: str = "zinb"

    ):
        super().__init__()
        if isinstance(n_hidden, int):
            n_hidden = [n_hidden]
        if len(n_hidden) > 1 and n_layers > 1:
            logger.warning(
                "Hidden node number list has been given, hidden layer number is ignored. " + \
                "Now hidden node number list is " + \
                str(n_hidden)
            )
        else:
            n_hidden = n_hidden * n_layers
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.reconstruction_loss = reconstruction_loss
        # Automatically deactivate if useless
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.resolution = resolution
        # self.covariance_type = covariance_type
        #
        # if self.covariance_type not in ['fixed', 'spherical', 'diag']:
        #     raise ValueError("Invalid value for 'covariance_type': %s "
        #                      "'covariance_type' should be in "
        #                      "['fixed', 'spherical', 'diag']"
        #                      % self.covariance_type)

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))  # 参数dispersion, size: (n_input)
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch',"
                " 'gene-label', 'gene-cell'], but input was "
                "{}.format(self.dispersion)"
            )

        # if self.covariance_type == "diag":
        #     self.register_buffer('logvar', torch.zeros(n_labels, n_latent))
        # elif self.covariance_type == "spherical":
        #     self.register_buffer('logvar', torch.zeros(n_labels, 1))

        self.classifier = self.classify

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        self.z_encoder = EncoderList(
            n_input,
            n_latent,
            n_hidden=n_hidden,
            n_cat_list=[n_batch],
            dropout_rate=dropout_rate
        )
        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = EncoderList(
            n_input, 1, n_hidden=n_hidden, n_cat_list=[n_batch], dropout_rate=dropout_rate,
        )
        # decoder goes from n_latent-dimensional space to n_input-d data
        self.decoder = DecoderList(
            n_latent,
            n_input,
            n_hidden=n_hidden[::-1],
            n_cat_list=[n_batch],
        )

        self.pretrain = True

    def classify(self, z):
        cs, zs = broadcast_labels(None, z, n_broadcast=self.n_labels)
        pz_mu_cs = torch.mm(cs, self.mu)
        # if self.covariance_type == "diag":
        #     pz_var_cs = torch.mm(cs, torch.exp(self.logvar)) + 1e-4
        # elif self.covariance_type == "spherical":
        #     logvar = self.logvar.repeat(1, self.n_latent)
        #     pz_var_cs = torch.mm(cs, torch.exp(logvar)) + 1e-4
        # else:
        pz_var_cs = torch.ones_like(pz_mu_cs)
        log_pz_c = Normal(pz_mu_cs, torch.sqrt(pz_var_cs)).log_prob(zs).sum(dim=-1).view(self.n_labels,
                                                                                         -1).t() + torch.log(
            self.pi + 1e-8)
        qc_z = F.softmax(log_pz_c, dim=-1)
        # log_pz_c = 1 / (1 + (zs - pz_mu_cs).pow(2).sum(dim=-1)).view(self.n_labels, -1).t() * self.pi
        # qc_z = log_pz_c / log_pz_c.sum(dim=-1, keepdim=True)

        return qc_z

    def sample_from_posterior_z(
            self, x, batch_index=None, give_mean=False
    ):
        """Samples the tensor of latent values from the posterior

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param give_mean: is True when we want the mean of the posterior  distribution rather than sampling
        :param n_samples: how many MC samples to average over for transformed mean
        :return: tensor of shape ``(batch_size, n_latent)``
        """
        if self.log_variational:
            x = torch.log(1 + x)
        qz_m, qz_v, z = self.z_encoder(x, batch_index)
        if give_mean:
            z = qz_m
        qc = self.classifier(z)
        return z, qc

    def get_reconstruction_loss(
            self, x, px_rate, px_r, px_dropout, **kwargs
    ) -> torch.Tensor:
        """Return the reconstruction loss (for a minibatch)
        """
        # Reconstruction Loss
        if self.reconstruction_loss == "zinb":
            reconst_loss = -log_zinb_positive(x, px_rate, px_r, px_dropout).sum(dim=-1)
        elif self.reconstruction_loss == "nb":
            reconst_loss = -log_nb_positive(x, px_rate, px_r).sum(dim=-1)
        return reconst_loss

    def get_sample_rate(
            self, x, batch_index=None, n_samples=1, transform_batch=None
    ) -> torch.Tensor:
        """Returns the tensor of means of the negative binomial distribution

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param n_samples: number of samples
        :param transform_batch: int of batch to transform samples into
        :return: tensor of means of the negative binomial distribution with shape ``(batch_size, n_input)``
        """
        return self.inference(
            x,
            batch_index=batch_index,
            n_samples=n_samples,
            transform_batch=transform_batch,
        )["px_rate"]

    def get_sample_scale_from_z(self, z, batch_index):

        px = self.decoder.px_decoder(z, batch_index)
        px_scale = self.decoder.px_scale_decoder(px)
        return px_scale

    def get_sample_scale(self, x, batch_index=None, n_samples=1, transform_batch=None
                         ) -> torch.Tensor:
        return self.inference(
            x,
            batch_index=batch_index,
            n_samples=n_samples,
            transform_batch=transform_batch,
        )["px_scale"]

    def inference(
            self, x, batch_index=None, n_samples=1, transform_batch=None
    ) -> Dict[str, torch.Tensor]:
        """Helper function used in forward pass
        """
        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)

        # Sampling
        qz_m, qz_v, z = self.z_encoder(x_, batch_index)
        ql_m, ql_v, library = self.l_encoder(x_, batch_index)

        # qc = self.classifier(z)
        # qc_square = qc ** 2
        # qc = qc_square / qc_square.sum(dim=-1, keepdim=True)

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            # when z is normal, untran_z == z
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.z_encoder.z_transformation(untran_z)
            ql_m = ql_m.unsqueeze(0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
            ql_v = ql_v.unsqueeze(0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
            library = Normal(ql_m, ql_v.sqrt()).sample()

        if transform_batch is not None:
            dec_batch_index = transform_batch * torch.ones_like(batch_index)
        else:
            dec_batch_index = batch_index

        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion, z, library, dec_batch_index
        )
        if self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(dec_batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r
        px_r = torch.exp(px_r) + 1e-4

        return dict(
            px_scale=px_scale,
            px_r=px_r,
            px_rate=px_rate,
            px_dropout=px_dropout,
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
            # qc=qc,
            ql_m=ql_m,
            ql_v=ql_v,
            library=library,
        )

    def forward(
            self, x, local_l_mean, local_l_var, batch_index=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Returns the reconstruction loss and the KL divergences

        :param x: tensor of values with shape (batch_size, n_input)
        :param local_l_mean: tensor of means of the prior distribution of latent variable l
         with shape (batch_size, 1)
        :param local_l_var: tensor of variancess of the prior distribution of latent variable l
         with shape (batch_size, 1)
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param y: tensor of cell-types labels with shape (batch_size, n_labels)
        :return: the reconstruction loss and the Kullback divergences
        """
        # Parameters for z latent distribution
        outputs = self.inference(x, batch_index)
        qz_m = outputs["qz_m"]
        qz_v = outputs["qz_v"]
        z = outputs["z"]
        # qc = outputs["qc"]
        ql_m = outputs["ql_m"]
        ql_v = outputs["ql_v"]
        px_rate = outputs["px_rate"]
        px_r = outputs["px_r"]
        px_dropout = outputs["px_dropout"]

        kl_divergence_l = kl(
            Normal(ql_m, torch.sqrt(ql_v)),
            Normal(local_l_mean, torch.sqrt(local_l_var)),
        ).sum(dim=-1)

        if self.pretrain:
            mean = torch.zeros_like(qz_m)
            scale = torch.ones_like(qz_v)
            kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(
                dim=1)
            kl_divergence = kl_divergence_z
            reconst_loss = self.get_reconstruction_loss(x, px_rate, px_r, px_dropout) + kl_divergence_l
            # latent_variable = dict(z=z, qc=None)
        else:
            qc = self.classifier(z)
            c_prior = self.pi.expand(qc.size())
            kl_divergence_c = kl(Categorical(qc), Categorical(c_prior))
            cs, zs = broadcast_labels(None, z, n_broadcast=self.n_labels)

            pz_mu_cs = torch.mm(cs, self.mu)
            # if self.covariance_type == "diag":
            #     pz_var_cs = torch.mm(cs, torch.exp(self.logvar)) + 1e-4
            # elif self.covariance_type == "spherical":
            #     logvar = self.logvar.repeat(1, self.n_latent)
            #     pz_var_cs = torch.mm(cs, torch.exp(logvar)) + 1e-4
            # else:
            pz_var_cs = torch.ones_like(pz_mu_cs)

            loss_z_weight = Normal(qz_m, torch.sqrt(qz_v)).log_prob(z).sum(
                dim=-1)

            loss_z_unweight = -(Normal(pz_mu_cs, torch.sqrt(pz_var_cs)).log_prob(zs).sum(
                dim=-1).view(self.n_labels, -1).t() * qc
                                ).sum(dim=-1)
            kl_divergence = kl_divergence_l + kl_divergence_c
            reconst_loss = self.get_reconstruction_loss(x, px_rate, px_r, px_dropout) + loss_z_unweight + loss_z_weight

            # latent_variable = dict(z=z, qc=qc)

        # return reconst_loss, kl_divergence, 0.0, latent_variable
        return reconst_loss, kl_divergence, 0.0

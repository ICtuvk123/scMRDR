import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import warnings
# import torchvision.transforms as transforms
from .loss import *
from .diffusion import GaussianDiffusion1D, LatentDenoiserMLP
from torch.nn import functional as F
import scipy as sp
import ot

class ModalityDiscriminator(nn.Module):
    '''
    Discriminator for modality classification.
    Args:
        z_dim (int): Dimension of the input latent space.
        num_modalities (int): Number of modalities to classify.
        layer_dims (list): List of hidden layer dimensions.
        dropout_rate (float): Dropout rate for regularization.
    '''
    def __init__(self, z_dim, num_modalities, layer_dims=[128,128], dropout_rate=0.2):
        super(ModalityDiscriminator, self).__init__()
        layers = []
        current_dim = z_dim
        for dim in layer_dims:
            layers.append(nn.Linear(current_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = dim
        layers.append(nn.Linear(layer_dims[-1], num_modalities))
        self.model = nn.Sequential(*layers) 
        # self.model = nn.Sequential(
        #     nn.Linear(z_dim, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.LeakyReLU(0.1),
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(hidden_dim, num_modalities)
        # )    
    def forward(self, z):
        '''
        Forward pass through the discriminator.
        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, z_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_modalities).
        '''
        z = self.model(z)
        return z


class Encoder(nn.Module):
    '''
    Encoder for the VAE model.
    Args:
        device (torch.device): Device to run the model on.
        input_dim (int): Dimension of the input data.
        layer_dims (list): List of hidden layer dimensions.
        latent_dim (int): Dimension of the latent space.
        dropout_rate (float): Dropout rate for regularization.
    '''
    def __init__(self, device, input_dim = 3000, layer_dims = [500,100], latent_dim = 20,
                 dropout_rate = 0.5):
        super(Encoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # q(z|x)
        layers_zxy = []
        current_dim = input_dim
        for dim in layer_dims:
            layers_zxy.append(nn.Linear(current_dim, dim))
            layers_zxy.append(nn.BatchNorm1d(dim))
            layers_zxy.append(nn.LeakyReLU())
            layers_zxy.append(nn.Dropout(dropout_rate))
            current_dim = dim
        self.zxy_encoder = nn.Sequential(*layers_zxy) 
        
        self.mu_layer = nn.Linear(layer_dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(layer_dims[-1], latent_dim)
        
    def forward(self,x):
        '''
        Forward pass through the encoder.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        Returns:
            z (torch.Tensor): Latent variable tensor of shape (batch_size, latent_dim).
            mu (torch.Tensor): Mean of the latent variable distribution.
            logvar (torch.Tensor): Log variance of the latent variable distribution.
        '''
        h = self.zxy_encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar #
    
    def reparameterize(self, mu, logvar):
        '''
        Reparameterization trick to sample from the latent variable distribution.
        Args:
            mu (torch.Tensor): Mean of the latent variable distribution.
            logvar (torch.Tensor): Log variance of the latent variable distribution.
        Returns:
            z (torch.Tensor): Sampled latent variable tensor.
        '''
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

class MSEDecoder(nn.Module):
    '''
    MSE Decoder for the VAE model.
    Args:
        device (torch.device): Device to run the model on.
        input_dim (int): Dimension of the input data.
        covariate_dim (int): Dimension of the batch size.
        layer_dims (list): List of hidden layer dimensions.
        latent_dim (int): Dimension of the latent space.
        dropout_rate (float): Dropout rate for regularization.
    '''
    def __init__(self, device, input_dim = 3000, covariate_dim = 1, layer_dims = [500,100], latent_dim = 20,
                 dropout_rate = 0.5, positive_outputs=True):
        super(MSEDecoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.covariate_dim = covariate_dim
        
        # p(x|z,c)
        layers_xz = []
        current_dim =  latent_dim + covariate_dim
        for dim in reversed(layer_dims):
            layers_xz.append(nn.Linear(current_dim, dim))
            layers_xz.append(nn.LeakyReLU(0.1))
            layers_xz.append(nn.BatchNorm1d(dim))
            layers_xz.append(nn.Dropout(dropout_rate))
            current_dim = dim
        
        self.decoder = nn.Sequential(*layers_xz)   
        if positive_outputs: 
            self.mean_layer = nn.Sequential(nn.Linear(layer_dims[0], input_dim),
                                            nn.Softplus())
            # self.zero_inflation_rates = nn.Parameter(torch.ones(input_dim) * 0.5) 
        else:
            self.mean_layer = nn.Sequential(nn.Linear(layer_dims[0], input_dim)
                                            # nn.Softplus()
                                            )
        
    def forward(self,z,b):
        '''
        Forward pass through the decoder.
        Args:
            z (torch.Tensor): Latent variable tensor of shape (batch_size, latent_dim).
            b (torch.Tensor): Batch information tensor of shape (batch_size, covariate_dim).
        Returns:
            rho (torch.Tensor): Mean of the output distribution.
        '''
        if self.covariate_dim > 0:
            z = torch.cat([z, b],dim=1)
        h = self.decoder(z)
        rho = self.mean_layer(h) 
        return rho

class Decoder(nn.Module):
    '''
    ZINB Decoder for the VAE model.
    Args:
        device (torch.device): Device to run the model on.
        input_dim (int): Dimension of the input data.
        covariate_dim (int): Dimension of the batch size.
        modality_num (int): Number of modalities.
        layer_dims (list): List of hidden layer dimensions.
        latent_dim (int): Dimension of the latent space.
        dropout_rate (float): Dropout rate for regularization.
    '''
    def __init__(self, device, input_dim = 3000, covariate_dim = 1, modality_num=2, layer_dims = [500,100], latent_dim = 20,
                 dropout_rate = 0.5):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.covariate_dim = covariate_dim
        self.modality_num = modality_num
        
        # p(x|z,c)
        layers_xz = []
        current_dim =  latent_dim + covariate_dim
        for dim in reversed(layer_dims):
            layers_xz.append(nn.Linear(current_dim, dim))
            layers_xz.append(nn.BatchNorm1d(dim))
            layers_xz.append(nn.LeakyReLU(0.1))
            layers_xz.append(nn.Dropout(dropout_rate))
            current_dim = dim
        
        self.decoder = nn.Sequential(*layers_xz)    
        self.mean_layer = nn.Sequential(nn.Linear(layer_dims[0], input_dim),
                                        nn.Softmax(dim=-1))
        self.dispersion_layer = nn.Sequential(nn.Linear(layer_dims[0], input_dim),
                                              nn.Softplus()) # gene-cell-wise dispersion
        self.dispersion = nn.Parameter(torch.randn(input_dim)) # gene-wise dispersion
        # self.modality_flag = nn.Sequential(nn.Linear(modality_num,1),nn.Tanh())
        self.dispersion_modality = nn.Parameter(torch.randn(modality_num, input_dim)) 
        self.dropout_layer = nn.Sequential(
            nn.Linear(layer_dims[0], input_dim),
            nn.Sigmoid())
        
    def forward(self,z,b,m,dispersion_strategy="gene-modality"):
        '''
        Forward pass through the decoder.
        Args:  
            z (torch.Tensor): Latent variable tensor of shape (batch_size, latent_dim).
            b (torch.Tensor): Batch information tensor of shape (batch_size, covariate_dim).
            m (torch.Tensor): Modality information tensor of shape (batch_size, modality_num).
        Returns:
            rho (torch.Tensor): Mean of the output distribution.
            dispersion (torch.Tensor): Dispersion parameter of the output distribution.
            pi (torch.Tensor): Dropout probabilities for the output distribution.
        '''
        if self.covariate_dim > 0:
            z = torch.cat([z, b],dim=1)
        h = self.decoder(z)
        rho = self.mean_layer(h)  # Ensure positive outputs
        if dispersion_strategy == "gene":
            dispersion = torch.exp(self.dispersion)
        elif dispersion_strategy == "gene-modality":
            # dispersion = torch.outer(torch.squeeze(self.modality_flag(m)), self.dispersion) # N * G
            dispersion = m @ self.dispersion_modality    # N * G
            dispersion = torch.exp(dispersion) # Ensure positive outputs # gene-wise
        elif dispersion_strategy == "gene-cell":
            dispersion = self.dispersion_layer(h) # gene-cell wise
        pi = self.dropout_layer(h) 
        return rho, dispersion, pi

class NBDecoder(nn.Module):
    '''
    NB Decoder for the VAE model.
    Args:
        device (torch.device): Device to run the model on.
        input_dim (int): Dimension of the input data.
        covariate_dim (int): Dimension of the batch size.
        modality_num (int): Number of modalities.
        layer_dims (list): List of hidden layer dimensions.
        latent_dim (int): Dimension of the latent space.
        dropout_rate (float): Dropout rate for regularization.
    '''
    def __init__(self, device, input_dim = 3000, covariate_dim = 1, modality_num=2, layer_dims = [500,100], latent_dim = 20,
                 dropout_rate = 0.5):
        super(NBDecoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.covariate_dim = covariate_dim
        self.modality_num = modality_num
        
        # p(x|z,c)
        layers_xz = []
        current_dim =  latent_dim + covariate_dim
        for dim in reversed(layer_dims):
            layers_xz.append(nn.Linear(current_dim, dim))
            layers_xz.append(nn.BatchNorm1d(dim))
            layers_xz.append(nn.LeakyReLU(0.1))
            layers_xz.append(nn.Dropout(dropout_rate))
            current_dim = dim
        
        self.decoder = nn.Sequential(*layers_xz)    
        self.mean_layer = nn.Sequential(nn.Linear(layer_dims[0], input_dim),
                                        nn.Softmax(dim=-1))
        self.dispersion_layer = nn.Sequential(nn.Linear(layer_dims[0], input_dim),
                                              nn.Softplus()) # gene-cell-wise dispersion
        self.dispersion = nn.Parameter(torch.randn(input_dim)) # gene-wise dispersion
        # self.modality_flag = nn.Sequential(nn.Linear(modality_num,1),nn.Tanh())
        self.dispersion_modality = nn.Parameter(torch.randn(modality_num, input_dim)) 
        # self.dropout_layer = nn.Sequential(
        #     nn.Linear(layer_dims[0], input_dim),
        #     nn.Sigmoid())
        
    def forward(self,z,b,m,dispersion_strategy="gene-modality"):
        '''
        Forward pass through the decoder.
        Args:  
            z (torch.Tensor): Latent variable tensor of shape (batch_size, latent_dim).
            b (torch.Tensor): Batch information tensor of shape (batch_size, covariate_dim).
            m (torch.Tensor): Modality information tensor of shape (batch_size, modality_num).
        Returns:
            rho (torch.Tensor): Mean of the output distribution.
            dispersion (torch.Tensor): Dispersion parameter of the output distribution.
            pi (torch.Tensor): Dropout probabilities for the output distribution.
        '''
        if self.covariate_dim > 0:
            z = torch.cat([z, b],dim=1)
        h = self.decoder(z)
        rho = self.mean_layer(h)  # Ensure positive outputs
        if dispersion_strategy == "gene":
            dispersion = torch.exp(self.dispersion)
        elif dispersion_strategy == "gene-modality":
            # dispersion = torch.outer(torch.squeeze(self.modality_flag(m)), self.dispersion) # N * G
            dispersion = m @ self.dispersion_modality    # N * G
            dispersion = torch.exp(dispersion) # Ensure positive outputs # gene-wise
        elif dispersion_strategy == "gene-cell":
            dispersion = self.dispersion_layer(h) # gene-cell wise
        pi = torch.zeros_like(rho)
        return rho, dispersion, pi

class EmbeddingNet(nn.Module):
    '''
    Models to get the unified latent embeddings.
    Args:
        device (torch.device): Device to run the model on.
        input_dim (int): Dimension of the input data.
        modality_num (int): Number of modalities.
        covariate_dim (int): Dimension of the covariates (like sequencing batches).
        celltype_num (int): Dimension of the cell type information. Default is 0.
        layer_dims (list): List of hidden layer dimensions.
        latent_dim_shared (int): Dimension of the shared latent space.
        latent_dim_specific (int): Dimension of the modality-specific latent space.
        dropout_rate (float): Dropout rate for regularization.
        beta (float): Weight for the KL divergence term.
        gamma (float): Weight for the isometric loss term.
        lambda_adv (float): Weight for the adversarial loss term.
        feat_mask (torch.Tensor): Feature mask for the input data.
        distribution (str): Distribution of the data, can be "ZINB", "NB", "Normal", "Normal_positive".
        encoder_covariates (bool): Whether to include covariates in the encoder.
        eps (float): Small value to avoid division by zero in loss calculations.
    '''

    def __init__(self, device, input_dim, modality_num, covariate_dim = 1, celltype_num = 0,
                layer_dims=[500,100], latent_dim_shared=20,
                latent_dim_specific=20, dropout_rate = 0.5, beta = 2, gamma = 1, lambda_adv = 0.01,
                feat_mask = None, distribution = "ZINB", # count_data = True, positive_outputs = True,
                encoder_covariates=False, eps=1e-10,
                latent_backend="vae",
                lambda_prior_diff=1.0, diffusion_steps=200,
                diffusion_hidden_dim=512, diffusion_time_embed_dim=64,
                diffusion_beta_schedule="linear",
                diffusion_prior_cond="none",
                beta_specific=None,
                lambda_diff=None,
                diffusion_cond=None):
        super(EmbeddingNet, self).__init__()
        
        self.beta = beta
        self.device = device
        self.eps = eps
        # self.paired= paired
        self.input_dim = input_dim
        self.latent_dim_shared = latent_dim_shared
        self.latent_dim_specific = latent_dim_specific
        self.covariate_dim = covariate_dim
        self.modality_num = modality_num
        self.celltype_num = celltype_num
        self.encoder_covariates = encoder_covariates
        self.gamma = gamma
        self.lambda_adv = lambda_adv
        if beta_specific is None:
            beta_specific = beta
        self.beta_specific = beta_specific
        if lambda_diff is not None:
            warnings.warn(
                "lambda_diff is deprecated; use lambda_prior_diff instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            lambda_prior_diff = lambda_diff
        self.lambda_prior_diff = lambda_prior_diff
        self.latent_backend = latent_backend
        if diffusion_cond is not None:
            warnings.warn(
                "diffusion_cond is deprecated; use diffusion_prior_cond instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            diffusion_prior_cond = diffusion_cond
        self.diffusion_prior_cond = diffusion_prior_cond
        self.feat_mask = feat_mask.to(self.device)

        if self.latent_backend not in {"vae", "diffusion"}:
            raise ValueError("latent_backend must be 'vae' or 'diffusion'")
        if self.diffusion_prior_cond not in {"none", "modality", "modality_batch", "modality_batch_celltype"}:
            raise ValueError(
                "diffusion_prior_cond must be one of: none, modality, modality_batch, modality_batch_celltype"
            )

        self.distribution = distribution
        if self.distribution in ["ZINB", "NB"]:
            self.count_data = True
            self.positive_outputs = True
        elif self.distribution == "Normal":
            self.count_data = False
            self.positive_outputs = False
        elif self.distribution == "Normal_positive":
            self.count_data = False
            self.positive_outputs = True

        self.encoder_shared = Encoder(device, input_dim+covariate_dim*encoder_covariates+self.celltype_num, 
                                      layer_dims, latent_dim_shared, dropout_rate)
        self.encoder_specific = Encoder(device, input_dim+modality_num+covariate_dim*encoder_covariates+self.celltype_num, 
                                        layer_dims, latent_dim_specific, dropout_rate)
        if self.distribution == "ZINB":
            self.decoder = Decoder(device, input_dim, covariate_dim, modality_num, layer_dims, 
                                   latent_dim_shared+latent_dim_specific, dropout_rate)
        elif self.distribution == "NB":
            self.decoder = Decoder(device, input_dim, covariate_dim, modality_num, layer_dims, 
                                   latent_dim_shared+latent_dim_specific, dropout_rate)
        else:
            self.decoder = MSEDecoder(device, input_dim, covariate_dim, layer_dims, latent_dim_shared+latent_dim_specific, 
                                      dropout_rate, positive_outputs=self.positive_outputs)
            
        self.prior_net_specific = nn.Sequential(nn.Linear(modality_num + self.celltype_num, 10),
                                    nn.LeakyReLU(0.1),
                                    nn.Dropout(dropout_rate),
                                    nn.Linear(10, 2))
        self.discriminator = ModalityDiscriminator(latent_dim_shared, modality_num, layer_dims=layer_dims, dropout_rate=dropout_rate)   

        self.diffusion = None
        self.diffusion_denoiser = None
        if self.latent_backend == "diffusion":
            cond_dim = 0
            if self.diffusion_prior_cond != "none":
                cond_dim = self.modality_num
                if self.diffusion_prior_cond in {"modality_batch", "modality_batch_celltype"}:
                    cond_dim += self.covariate_dim
                if self.diffusion_prior_cond == "modality_batch_celltype":
                    cond_dim += self.celltype_num
            self.diffusion = GaussianDiffusion1D(
                num_steps=diffusion_steps,
                beta_schedule=diffusion_beta_schedule,
            )
            self.diffusion_denoiser = LatentDenoiserMLP(
                latent_dim=self.latent_dim_shared,
                cond_dim=cond_dim,
                hidden_dim=diffusion_hidden_dim,
                time_embed_dim=diffusion_time_embed_dim,
                dropout_rate=dropout_rate,
            )

    def _build_diffusion_cond(self, m, b, w):
        if self.diffusion_prior_cond == "none":
            return None
        cond_parts = [m]
        if self.diffusion_prior_cond in {"modality_batch", "modality_batch_celltype"} and self.covariate_dim > 0:
            cond_parts.append(b)
        if self.diffusion_prior_cond == "modality_batch_celltype" and self.celltype_num > 0:
            cond_parts.append(w)
        return torch.cat(cond_parts, dim=-1)

    def _encode_shared(self, x, b, w):
        if self.celltype_num == 0:
            if self.encoder_covariates:
                return self.encoder_shared(torch.cat([x, b], dim=-1))
            return self.encoder_shared(x)
        if self.encoder_covariates:
            return self.encoder_shared(torch.cat([x, b, w], dim=-1))
        return self.encoder_shared(torch.cat([x, w], dim=-1))

    def _encode_specific(self, x, m, b, w):
        if self.celltype_num == 0:
            if self.encoder_covariates:
                return self.encoder_specific(torch.cat([x, m, b], dim=-1))
            return self.encoder_specific(torch.cat([x, m], dim=-1))
        if self.encoder_covariates:
            return self.encoder_specific(torch.cat([x, m, b, w], dim=-1))
        return self.encoder_specific(torch.cat([x, m, w], dim=-1))

    def _specific_prior(self, m, w):
        if self.celltype_num == 0:
            return torch.chunk(self.prior_net_specific(m), 2, dim=-1)
        return torch.chunk(self.prior_net_specific(torch.cat([m, w], dim=-1)), 2, dim=-1)

    def _build_shared_latent(self, x, b, m, w):
        z_shared_raw, mu_shared_raw, logvar_shared_raw = self._encode_shared(x, b, w)
        if self.latent_backend == "vae":
            prior_diff_loss = torch.tensor(0.0, device=x.device)
            return z_shared_raw, mu_shared_raw, logvar_shared_raw, prior_diff_loss

        z0_shared = mu_shared_raw
        cond = self._build_diffusion_cond(m, b, w)
        t = self.diffusion.sample_timesteps(z0_shared.shape[0], z0_shared.device)
        noise = torch.randn_like(z0_shared)
        z_t = self.diffusion.q_sample(z0_shared, t, noise)
        eps_pred = self.diffusion_denoiser(z_t, t, cond)
        z_shared = self.diffusion.predict_x0_from_eps(z_t, t, eps_pred)
        prior_diff_loss = F.mse_loss(eps_pred, noise)
        mu_shared = z_shared
        logvar_shared = torch.zeros_like(z_shared)
        return z_shared, mu_shared, logvar_shared, prior_diff_loss

    def vae_parameters(self):
        modules = [
            self.encoder_shared,
            self.encoder_specific,
            self.decoder,
            self.prior_net_specific,
        ]
        if self.latent_backend == "diffusion":
            modules.append(self.diffusion_denoiser)
        for module in modules:
            yield from module.parameters()
    
    def forward(self,x,b,m,i,w,stage="vae",return_adv_components=False):
        '''
        Forward pass through the embedding network.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            b (torch.Tensor): Batch information tensor of shape (batch_size, covariate_dim).
            m (torch.Tensor): Modality information tensor of shape (batch_size, modality_num).
            i (torch.Tensor): Mask indicator tensor of shape (batch_size, input_dim).
            w (torch.Tensor): Cell type information tensor of shape (batch_size, celltype_num).
            stage (str): Stage of the model, can be "vae", "discriminator", or "warmup".
        Returns:
            mu_shared (torch.Tensor): Mean of the shared latent variable distribution.
            mu_specific (torch.Tensor): Mean of the specific latent variable distribution.
            total_loss (torch.Tensor): Total loss for the VAE model.
            loss_dict (dict): Dictionary containing individual loss components.
        '''
        if stage=="vae":
            x_original = x
            if self.count_data:
                x = torch.log1p(x)

            prior_mu, prior_logvar = self._specific_prior(m, w)
            z_shared, mu_shared, logvar_shared, prior_diff_loss = self._build_shared_latent(x, b, m, w)
            z_specific, mu_specific, logvar_specific = self._encode_specific(x, m, b, w)

            z = torch.cat([z_shared, z_specific], dim=-1)
            if self.count_data:
                rho,dispersion,pi = self.decoder(z, b, m)
                s = self.sample_sequencing_depth(x_original)
            else:
                rho = self.decoder(z, b)
    
            # rho2,dispersion2,pi2 = self.decoder2(z_shared, b, m)
            
            #loss
            if self.feat_mask is not None:
                mask = i @ self.feat_mask
            else:
                mask = None
            
            zinb_loss = ZINBLoss()
            if self.count_data:
                recon_loss = zinb_loss(x_original, rho, dispersion, pi, s, mask, eps = self.eps)
            else:
                if self.positive_outputs:
                    # recon_loss = ZeroInflatedMSELoss()(x_original, rho, self.decoder.zero_inflation_rates)
                    recon_loss = mseLoss(x_original, rho)
                else:
                    recon_loss = mseLoss(x_original, rho)
            kl_specific = klLoss_prior(mu_specific, logvar_specific, prior_mu, prior_logvar)
            kl_shared = torch.tensor(0.0, device=x.device)
            if self.latent_backend == "vae":
                kl_shared = klLoss(mu_shared, logvar_shared)
            kl_z = kl_specific + kl_shared
            # preserve_loss = zinb_loss(x_original, rho2, dispersion2, pi2, s, eps = self.eps)
            preserve_loss = isometric_loss(torch.cat([mu_shared, mu_specific],dim=-1),mu_shared,m)
            # preserve_loss = isometric_loss(torch.cat([z_shared, z_specific],dim=-1),z_shared,m)
            # preserve_loss = sammon_loss(torch.cat([z_shared, z_specific],dim=-1),z_shared,m)
            # preserve_loss = laplacian_loss(torch.cat([z_shared, z_specific],dim=-1),z_shared,m)
            # preserve_loss = frobenius_isometric_loss(torch.cat([z_shared, z_specific],dim=-1),z_shared,m)
            # preserve_loss = knn_structure_loss(torch.cat([mu_shared, mu_specific],dim=-1),mu_shared,m)
            #z_per_class = [z_shared[m[:, i].bool()] for i in range(m.shape[1])]
            #align_loss = torch.stack([MMD(z_per_class[0], z_per_class[i]) for i in range(1, len(z_per_class))]).sum()

            modality_labels = torch.argmax(m, dim=1)
            modality_logits_adv = self.discriminator(z_shared)  # gradients allowed here

            if return_adv_components:
                per_sample_adv = -F.cross_entropy(modality_logits_adv, modality_labels, reduction='none')  # (B,)
                adv_loss_scalar = per_sample_adv.sum() / m.shape[0]
                base_loss = (
                    recon_loss
                    + self.beta_specific * kl_specific
                    + self.beta * kl_shared
                    + self.gamma * preserve_loss
                    + self.lambda_prior_diff * prior_diff_loss
                )
                loss_dict = {'total_loss': (base_loss + self.lambda_adv * adv_loss_scalar).item(),
                            'recon_loss': recon_loss.item(), 'kl_z': kl_z.item(),
                            'kl_specific': kl_specific.item(),
                            'kl_shared': kl_shared.item(),
                            'preserve_loss': preserve_loss.item(),
                            'adv_loss': adv_loss_scalar.item(),
                            'prior_diff_loss': prior_diff_loss.item(),
                            'diff_loss': prior_diff_loss.item(),
                            '_z_shared': z_shared,
                            '_modality_logits': modality_logits_adv,
                            '_per_sample_adv': per_sample_adv,
                            }
                return mu_shared, mu_specific, base_loss, loss_dict
            else:
                adv_loss = -F.cross_entropy(modality_logits_adv, modality_labels, reduction='sum')/m.shape[0]
                total_loss = (
                    recon_loss
                    + self.beta_specific * kl_specific
                    + self.beta * kl_shared
                    + self.gamma * preserve_loss
                    + self.lambda_prior_diff * prior_diff_loss
                    + self.lambda_adv * adv_loss
                )
                loss_dict = {'total_loss':total_loss.item(),
                            'recon_loss':recon_loss.item(),'kl_z':kl_z.item(),
                            'kl_specific': kl_specific.item(),
                            'kl_shared': kl_shared.item(),
                            'preserve_loss': preserve_loss.item(),
                            'adv_loss': adv_loss.item(),
                            'prior_diff_loss': prior_diff_loss.item(),
                            'diff_loss': prior_diff_loss.item(),
                            }
                return mu_shared, mu_specific, total_loss, loss_dict
        elif stage=="discriminator":
            if self.count_data:
                x = torch.log1p(x)

            z_shared, mu_shared, _ = self._encode_shared(x, b, w)
            if self.latent_backend == "diffusion":
                z_for_disc = mu_shared
            else:
                z_for_disc = z_shared

            modality_labels = torch.argmax(m, dim=1)
            z_shared_detached = z_for_disc.clone().detach()
            modality_logits = self.discriminator(z_shared_detached) #
            discri_loss = F.cross_entropy(modality_logits, modality_labels, reduction='sum')/m.shape[0]
            return discri_loss
        
        elif stage=="warmup":
            x_original = x
            if self.count_data:
                x = torch.log1p(x)

            prior_mu, prior_logvar = self._specific_prior(m, w)
            z_shared, mu_shared, logvar_shared, prior_diff_loss = self._build_shared_latent(x, b, m, w)
            z_specific, mu_specific, logvar_specific = self._encode_specific(x, m, b, w)
            z = torch.cat([z_shared, z_specific], dim=-1)
            if self.count_data:
                rho,dispersion,pi = self.decoder(z, b, m)
                s = self.sample_sequencing_depth(x_original)
            else:
                rho = self.decoder(z, b)
            
            # loss
            if self.feat_mask is not None:
                mask = m @ self.feat_mask
            else:
                mask = None
            
            zinb_loss = ZINBLoss()
            if self.count_data:
                recon_loss = zinb_loss(x_original, rho, dispersion, pi, s, mask, eps = self.eps)
            else:
                recon_loss = mseLoss(x_original, rho, mask)
            kl_specific = klLoss_prior(mu_specific, logvar_specific, prior_mu, prior_logvar)
            kl_shared = torch.tensor(0.0, device=x.device)
            if self.latent_backend == "vae":
                kl_shared = klLoss(mu_shared, logvar_shared)
            kl_z = kl_specific + kl_shared
            preserve_loss = isometric_loss(torch.cat([mu_shared, mu_specific],dim=-1),mu_shared,m)  
            # hsic = 1000 * HSICloss(z_shared,m)
            total_loss = (
                recon_loss
                + self.beta_specific * kl_specific
                + self.beta * kl_shared
                + self.gamma * preserve_loss
                + self.lambda_prior_diff * prior_diff_loss
            ) #+ hsic
            loss_dict = {'recon_loss':recon_loss.item(),'kl_z':kl_z.item(),
                        'kl_specific': kl_specific.item(),
                        'kl_shared': kl_shared.item(),
                        'preserve_loss': preserve_loss.item(),
                        'prior_diff_loss': prior_diff_loss.item(),
                        'diff_loss': prior_diff_loss.item(),
                        # 'hsic': hsic.item(),
                        'total_loss':total_loss.item(),
                        '_z_shared': z_shared,
                        }
            return mu_shared, mu_specific, total_loss, loss_dict
    
    def sample_sequencing_depth(self, x, strategy="observed"):
        '''
        Sample sequencing depth based on the strategy.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            strategy (str): Strategy for sampling sequencing depth, can be "batch_sample" or "observed".
        Returns:
            s (torch.Tensor): Sampled sequencing depth tensor of shape (batch_size, 1).
        '''
        if strategy=="batch_sample": # batch-wise empirically sample
            mu_s = torch.log(x.sum(dim=1) + 1.0).mean()
            sigma_s = torch.log(x.sum(dim=1) + 1.0).std()
            log_s = mu_s + sigma_s * torch.randn_like(sigma_s)
            s = torch.exp(log_s)
            # s = s.detach()
        elif strategy == "observed": # directly observed
            log_s = torch.log(x.sum(dim=1)).unsqueeze(1)
            s = torch.exp(log_s)
            # s = s.detach()
        return s
    
    def reparameterize(self, mu, logvar):
        '''
        Reparameterization trick to sample from the latent variable distribution.
        Args:
            mu (torch.Tensor): Mean of the latent variable distribution.
            logvar (torch.Tensor): Log variance of the latent variable distribution.
        Returns:
            z (torch.Tensor): Sampled latent variable tensor.
        '''
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    

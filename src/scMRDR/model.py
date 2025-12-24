import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
# import torchvision.transforms as transforms
from .loss import *
from torch.nn import functional as F
import scipy as sp
import ot
from einops import rearrange

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

def create_activation(name):
    if name == "gelu":
        return nn.GELU()
    elif name == "relu":
        return nn.ReLU()
    elif name == "mish":
        return Mish()
    return nn.Identity()

def create_norm(name, dim):
    if name == "layernorm":
        return nn.LayerNorm(dim)
    elif name == "batchnorm":
        return nn.BatchNorm1d(dim)
    return nn.Identity()

def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

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


class DisentanglementEncoder(nn.Module):
    def __init__(self, 
                 profile_size, 
                 out_dim, 
                 num_factor, 
                 label_categories,
                 bias = False,
                 out_act = "gelu",  
                 gamma = 35
                 ):
        super().__init__()
        if isinstance(out_act, str) or out_act is None:
            out_act = create_activation(out_act)
        self.num_factor = num_factor
        self.out_dim = out_dim
        self.profile_size = profile_size
        self.exogenous_encoder_m_v = nn.Sequential(
            nn.Linear(profile_size, profile_size // 4), 
            Mish(), 
            nn.Linear(profile_size // 4, num_factor * out_dim * 2)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_factor))
        else:
            self.register_parameter('bias', None)

        self.label_predictor = nn.ModuleList()
        for idx, num in enumerate(label_categories):
            self.label_predictor.append(nn.Sequential(
                nn.Linear(out_dim, num),
                nn.Softmax(dim = 1) 
            )
            )

        self.multilabelmulticate_loss = nn.CrossEntropyLoss()

        # discriminator for o and v
        self.discriminator_ov = nn.Linear(out_dim, 1)
        self.discriminator_ov2 = nn.Linear(num_factor, 1)
        self.discriminator_ov_act = nn.Sigmoid()
        
        self.gamma = gamma
    
    def normal_kl(self, mean1, logvar1, mean2, logvar2):
        """
        Compute the KL divergence between two gaussians.
        """
        tensor = None
        for obj in (mean1, logvar1, mean2, logvar2):
            if isinstance(obj, torch.Tensor):
                tensor = obj
                break
        assert tensor is not None, "at least one argument must be a Tensor"

        logvar1, logvar2 = [
            x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
            for x in (logvar1, logvar2)
        ]

        return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
        )

    def calculat_prior_kl(self, mean, log_var):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        """
        batch_size = mean.shape[0]
        kl_prior = self.normal_kl(
            mean1=mean, logvar1=log_var, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def sample(self, mean, log_var):
        noise = torch.randn_like(mean)
        return mean + (0.5 * log_var).exp() * noise
    
    def forward(self, x, o):
        exogenous_factor_m, exogenous_factor_v = torch.split(self.exogenous_encoder_m_v(x), self.num_factor * self.out_dim, dim=-1)
        prior_kl = self.calculat_prior_kl(exogenous_factor_m, exogenous_factor_v).mean()
        
        exogenous_factor = self.sample(exogenous_factor_m, exogenous_factor_v)
        exogenous_embs = rearrange(exogenous_factor, 'b (h d) -> b h d', h=self.num_factor)

        z = exogenous_embs
        concept_embs = z

        mask_recon_loss = torch.tensor(0.0, device=x.device)
        
        pred_o = []
        for idx, predictor in enumerate(self.label_predictor):
            pred_o.append(predictor(concept_embs[:,idx,:]))
        pred_o_loss = 0
        for idx, pred_o_idx in enumerate(pred_o):
            pred_o_loss_idx = self.multilabelmulticate_loss(pred_o_idx, o[:,idx])
            pred_o_loss += pred_o_loss_idx
        
        # take mean-level loss
        pred_o_loss /= (len(pred_o) + 1e-8)
        
        # new adversirial part
        pred_u = []
        for idx, predictor in enumerate(self.label_predictor):
            pred_u.append(predictor(concept_embs[:, -1, :]))
        pred_u_loss = 0
        for idx, pred_u_idx in enumerate(pred_u):
            pred_u_loss_idx = self.multilabelmulticate_loss(pred_u_idx, o[:,idx])
            pred_u_loss += pred_u_loss_idx
        # take mean-level loss
        discriminator_loss = - pred_u_loss / (len(pred_u) + 1e-8)
        
        return concept_embs, mask_recon_loss, pred_o_loss, discriminator_loss, prior_kl
    
    def extract_exogenous_embs(self, x):
        with torch.no_grad():
            exogenous_factor_m, exogenous_factor_v = torch.split(self.exogenous_encoder_m_v.eval()(x), self.num_factor * self.out_dim, dim=-1)
            exogenous_factor = self.sample(exogenous_factor_m, exogenous_factor_v)
            exogenous_embs = rearrange(exogenous_factor, 'b (h d) -> b h d', h=self.num_factor)
        return exogenous_embs



class Encoder(nn.Module):
    '''
    # Encoder for the VAE model.
    # Args:
    #     device (torch.device): Device to run the model on.
    #     input_dim (int): Dimension of the input data.
    #     layer_dims (list): List of hidden layer dimensions.
    #     latent_dim (int): Dimension of the latent space.
    #     dropout_rate (float): Dropout rate for regularization.
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
        # Forward pass through the encoder.
        # Args:
        #     x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        # Returns:
        #     z (torch.Tensor): Latent variable tensor of shape (batch_size, latent_dim).
        #     mu (torch.Tensor): Mean of the latent variable distribution.
        #     logvar (torch.Tensor): Log variance of the latent variable distribution.
        '''
        h = self.zxy_encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar #
    
    def reparameterize(self, mu, logvar):
        # Reparameterization trick to sample from the latent variable distribution.
        # Args:
        #     mu (torch.Tensor): Mean of the latent variable distribution.
        #     logvar (torch.Tensor): Log variance of the latent variable distribution.
        # Returns:
        #     z (torch.Tensor): Sampled latent variable tensor
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

class CrossAttention(nn.Module):
    def __init__(self,
                 query_dim, 
                 context_dim, 
                 heads = 8, 
                 dim_head = 64, 
                 dropout = 0., 
                 qkv_bias = False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        
        self.scale = dim_head ** -0.5
        self.heads = heads
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias = qkv_bias)
        self.to_k = nn.Linear(context_dim, inner_dim, bias = qkv_bias)
        self.to_v = nn.Linear(context_dim, inner_dim, bias = qkv_bias)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, *, context = None, mask = None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        if exists(mask):
            mnv = max_neg_value(sim) - torch.finfo(sim.dtype).max
            if sim.shape[1:] == sim.shape[1:]:
                mask = repeat(mask, 'b ... -> (b h) ...', h = h)
            else:
                mask = rearrange(mask, 'b ... -> b (...)')
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, mnv)
        
        attn = sim.softmax(dim = -1)
        # print(attn)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int, 
        d_head: int = 64, 
        self_attn: bool = False,
        cross_attn: bool = True,
        ts_cross_attn: bool = False, 
        final_act: Optional[nn.Module] = None,
        dropout: float = 0, 
        context_dim: Optional[int] = None, 
        gated_ff: bool = True, 
        checkpoint: bool = False,
        qkv_bias: bool = False, 
        linear_attn: bool = False, 
    ):
        super().__init__()
        assert self_attn or cross_attn, 'At least on attention layer'
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.ff = FeedForward(dim, dropout=dropout, glu = gated_ff)
        if ts_cross_attn:
            raise NotImplementedError("Deprecated, please remove.")  # FIX: remove ts_cross_attn option
        else:
            assert not linear_attn, "Performer attention not setup yet."  # FIX: remove linear_attn option
            attn_cls = CrossAttention
        
        if self.cross_attn:
            self.attn1 = attn_cls(
                query_dim = dim, 
                context_dim = context_dim, 
                heads = n_heads, 
                dim_head = d_head, 
                dropout = dropout, 
                qkv_bias = qkv_bias
            )
        if self.self_attn:
            self.attn2 = attn_cls(
                query_dim = dim, 
                heads = n_heads, 
                dim_head = d_head, 
                dropout = dropout, 
                qkv_bias = qkv_bias
            )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.act = final_act
        self.checkpoint = checkpoint
        assert not self.checkpoint, "Checkpointing not available yet"
    
    @BatchedOperation(batch_dim=0, plain_num_dim=2)
    def forward(self, x, context=None, cross_mask=None, self_mask=None, **kwargs):
        if self.cross_attn:
            x = self.attn1(self.norm1(x), context=context, mask=cross_mask, **kwargs) + x
        if self.self_attn:
            x = self.attn2(self.norm2(x), mask=self_mask, **kwargs) + x
        x = self.ff(self.norm3(x)) + x
        if self.act is not None:
            x = self.act(x)
        return x


class Denoise_net(nn.Module):
    def __init__(self, 
                 dim, 
                 out_dim, 
                 depth = 4,
                 num_heads = 4, 
                 dim_head = 64,
                 dropout = 0., 
                 norm_type = "layernorm", 
                 num_layers = 1, 
                 act = 'gelu', 
                 out_act = None, 
                 with_time_emb = True):
        super().__init__()
        if isinstance(act, str) or act is None:
            act = create_activation(act)
        if isinstance(out_act, str) or out_act is None:
            out_act = create_activation(out_act)
        
        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim), 
                nn.Linear(dim, dim * 4), 
                Mish(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(nn.Sequential(
                nn.Linear(dim, dim),
                act,
                create_norm(norm_type, dim),
                nn.Dropout(dropout)
            ))
        self.layers.append(nn.Sequential(nn.Linear(dim, out_dim), out_act))
        
        # the embeddings will be given by encoder during the whole training part

        self.Cross_attention_module = nn.ModuleList([
            BasicTransformerBlock(out_dim, num_heads, dim_head, self_attn=False, cross_attn=True, context_dim=32, 
                                  qkv_bias=True, dropout=dropout, final_act=None)
            for _ in range(depth)
        ])
        self.decoder_norm = create_norm(norm_type, out_dim)
        
    def forward(self, x, x_start, time, embeddings, labels=None):
        # if self.cond_embed is not None:
        #     cond_emb = self.cond_embed(conditions)[0]
        #     x = x + cond_emb.squeeze(1)

        # if labels is not None and concept_embs is None:

        t = self.time_mlp(time) if exists(self.time_mlp) else None
        x = x + t
        x = x.unsqueeze(1)
        for blk in self.Cross_attention_module:
            x = blk(x = x, context = embeddings)
        x.squeeze_(1)
        x = self.decoder_norm(x)
        for layer in self.layers:
            x = layer(x)
        # x = self.layers(x)
        
        # return x, mask_recon_loss, pred_o_loss, discriminator_loss, prior_kl

        return x
    
        # elif labels is None and concept_embs is None:
        #     # print("No cross attention generation")
        #     t = self.time_mlp(time) if exists(self.time_mlp) else None
        #     x = x + t
        #     x = self.decoder_norm(x)
        #     for layer in self.layers:
        #         x = layer(x)
        #     # x = self.layers(x)
        #     return x
        
        # elif labels is None and concept_embs is not None:
        #     t = self.time_mlp(time) if exists(self.time_mlp) else None
        #     x = x + t
        #     x = x.unsqueeze(1)
        #     for blk in self.Cross_attention_module:
        #         x = blk(x = x, context = concept_embs)
        #     x.squeeze_(1)
        #     x = self.decoder_norm(x)
        #     for layer in self.layers:
        #         x = layer(x)
        #     # x = self.layers(x)
        #     return x
        
        # else:
        #     print("No condition for labels and factor embs all exisits")
        #     return

class GaussianDiffusion(nn.Module):
    def __init__(self, 
                 denosie_fn, 
                 *, 
                 profile_size, 
                #  channels = 3, 
                 timesteps = 1000, 
                 loss_type = "l1", 
                 betas = None):
        super().__init__()
        self.profile_size = profile_size
        self.denosie_fn = denosie_fn
        
     
        
        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = make_beta_schedule("linear", timesteps)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)   
          
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

 
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        self.loss_type = loss_type
        
        to_torch = partial(torch.tensor, dtype=torch.float32)
        
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
    
    def q_mean_variance(self, x_start, t):
        """
        Given x_0 and t, output x_t by adding noise
        """
        mean = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract_into_tensor(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    
    def predict_start_from_noise(self, x_t, t, noise):
        """
        
        """
        assert x_t.shape == noise.shape, "Please check the code and data"
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - 
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
                )
    
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start + 
            extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, x, t, clip_denoised: bool):
        # x_recon = self.predict_start_from_noise(x, t=t, noise=self.denosie_fn(x, t))
        x_recon = self.denosie_fn(x, t)
        # this should be setted as the data distribution
        if clip_denoised:
            x_recon.clamp_(0)
            # x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
    
    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=False, repeat_noise = False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # noise = default(noise, lambda: torch.randn_like(x_start))
        
        # no noise when t==0
        nonzero_mask = (1 - (t==0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    
    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = self.betas.device
        
        b = shape[0]
        img = torch.randn(shape, device = device)
        
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img
    
    @torch.no_grad()
    def sample(self, batch_size=16):
        profile_size = self.profile_size
        return self.p_sample_loop((batch_size, profile_size))

    def p_mean_variance_with_factor(self, x, t, concept_embs, clip_denoised: bool, eps = False):
        
        x_start = None
        if eps:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.denosie_fn(x, x_start, t, concept_embs = concept_embs))
        else:
            x_recon = self.denosie_fn(x, x_start, t, concept_embs = concept_embs)
            
        # this should be setted as the data distribution
        if clip_denoised:
            x_recon.clamp_(0)
            # x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_with_factor(self, x, t, concept_embs, clip_denoised=False, repeat_noise = False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance_with_factor(x=x, t=t, concept_embs=concept_embs, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # noise = default(noise, lambda: torch.randn_like(x_start))
        
        # no noise when t==0
        nonzero_mask = (1 - (t==0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop_with_factor(self, shape, concept_embs):
        device = self.betas.device
        
        b = shape[0]
        img = torch.randn(shape, device = device)
        
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            img = self.p_sample_with_factor(img, torch.full((b,), i, device=device, dtype=torch.long), concept_embs)
        return img

    @torch.no_grad()
    def sample_with_factor(self, concept_embs, batch_size=16):
        profile_size = self.profile_size
        return self.p_sample_loop_with_factor((batch_size, profile_size), concept_embs)
    
    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)
        
        assert x1.shape == x2.shape
        
        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))
        
        img = (1 - lam) * xt1 + lam *xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total = t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        
        return img

    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +  
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
                )
            
    def p_losses(self, x_start, embeddings, t, labels, weights, noise = None, eps = False):
        b, c = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))
        
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon, mask_recon_loss, pred_o_loss, discriminator_loss, prior_kl = self.denosie_fn(x_noisy, x_start, t, labels)
        
        assert x_recon.shape == x_noisy.shape, "Please check the code and data"
        
        if self.loss_type == "l1":
            if eps:
                loss = (((noise - x_recon).abs()) * weights[:, None]).sum()
            else:
                loss = (((x_start - x_recon).abs()) * weights[:, None]).sum()
        elif self.loss_type == "l2":
            if eps:
                loss = (((noise - x_recon)**2) * weights[:, None]).sum()
            else:
                loss = (((x_start - x_recon)**2) * weights[:, None]).sum()
        else:
            raise NotImplementedError()
        
        return loss, mask_recon_loss, pred_o_loss, discriminator_loss, prior_kl
    
    def forward(self, x, *args, **kwargs):
        b, c, device, profile_size, = *x.shape, x.device, self.profile_size
        assert c == profile_size, f'dimension of gene expression profile must be {profile_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, *args, **kwargs)


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
                encoder_covariates=False, eps=1e-10):
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
        self.feat_mask = feat_mask.to(self.device)

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
    
    def forward(self,x,b,m,i,w,stage="vae"):
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
            
            if self.celltype_num == 0:
                prior_mu, prior_logvar = torch.chunk(self.prior_net_specific(m), 2, dim=-1)
                if self.encoder_covariates:
                    z_shared, mu_shared, logvar_shared = self.encoder_shared(torch.cat([x,b],dim=-1))
                    z_specific, mu_specific, logvar_specific = self.encoder_specific(torch.cat([x,m,b],dim=-1))
                else:
                    z_shared, mu_shared, logvar_shared = self.encoder_shared(x)
                    z_specific, mu_specific, logvar_specific = self.encoder_specific(torch.cat([x,m],dim=-1))
            else:
                prior_mu, prior_logvar = torch.chunk(self.prior_net_specific(torch.cat([m,w],dim=-1)), 2, dim=-1)
                if self.encoder_covariates:
                    z_shared, mu_shared, logvar_shared = self.encoder_shared(torch.cat([x,b,w],dim=-1))
                    z_specific, mu_specific, logvar_specific = self.encoder_specific(torch.cat([x,m,b,w],dim=-1))
                else:
                    z_shared, mu_shared, logvar_shared = self.encoder_shared(torch.cat([x,w],dim=-1))
                    z_specific, mu_specific, logvar_specific = self.encoder_specific(torch.cat([x,m,w],dim=-1))
            
            # concat z_shared adn z_specific to predict q(x|z)
            z = torch.cat([z_shared,z_specific],dim=-1)

            X_dim = input_dim+modality_num+covariate_dim*encoder_covariates+self.celltype_num
            Denoise_model = Denoise_net(X_dim,X_dim)
            diffusion_model = GaussianDiffusion(Denoise_model,X_dim,timesteps,loss_type,betas)

            diffusion_model(torch.cat([x,b,w],dim=-1),z,t,labels,weights)
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
            kl_z = klLoss_prior(mu_specific, logvar_specific, prior_mu, prior_logvar)+\
                klLoss(mu_shared, logvar_shared)
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
            adv_loss = -F.cross_entropy(modality_logits_adv, modality_labels, reduction='sum')/m.shape[0]

            # z_shared_detached = z_shared.clone().detach()
            # modality_logits = self.discriminator(z_shared_detached) #
            # discri_loss = F.cross_entropy(modality_logits, modality_labels, reduction='sum')/m.shape[0]

                
            total_loss = recon_loss + self.beta*kl_z + self.gamma * preserve_loss + self.lambda_adv * adv_loss # + align_loss
            loss_dict = {'total_loss':total_loss.item(), 
                        'recon_loss':recon_loss.item(),'kl_z':kl_z.item(),
                        'preserve_loss': preserve_loss.item(), #,'align_loss':align_loss.item()
                        'adv_loss': adv_loss.item()
                        } 
            return mu_shared, mu_specific, total_loss, loss_dict
        elif stage=="discriminator":
            x_original = x
            if self.count_data:
                x = torch.log1p(x)
            
            # prior_mu, prior_logvar = torch.chunk(self.prior_net_specific(m), 2, dim=-1)
            if self.encoder_covariates:
                z_shared, mu_shared, logvar_shared = self.encoder_shared(torch.cat([x,b],dim=-1))
                # z_specific, mu_specific, logvar_specific = self.encoder_specific(torch.cat([x,m,b],dim=-1))
            else:
                z_shared, mu_shared, logvar_shared = self.encoder_shared(x)
                # z_specific, mu_specific, logvar_specific = self.encoder_specific(torch.cat([x,m],dim=-1))
    
            # rho2,dispersion2,pi2 = self.decoder2(z_shared, b, m)
            
            #loss
            modality_labels = torch.argmax(m, dim=1)
            z_shared_detached = z_shared.clone().detach()
            modality_logits = self.discriminator(z_shared_detached) #
            discri_loss = F.cross_entropy(modality_logits, modality_labels, reduction='sum')/m.shape[0]
            return discri_loss
        
        elif stage=="warmup":
            x_original = x
            if self.count_data:
                x = torch.log1p(x)
            
            prior_mu, prior_logvar = torch.chunk(self.prior_net_specific(m), 2, dim=-1)
            if self.encoder_covariates:
                z_shared, mu_shared, logvar_shared = self.encoder_shared(torch.cat([x,b],dim=-1))
                z_specific, mu_specific, logvar_specific = self.encoder_specific(torch.cat([x,m,b],dim=-1))
            else:
                z_shared, mu_shared, logvar_shared = self.encoder_shared(x)
                z_specific, mu_specific, logvar_specific = self.encoder_specific(torch.cat([x,m],dim=-1))
            z = torch.cat([z_shared,z_specific],dim=-1)
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
            kl_z = klLoss_prior(mu_specific, logvar_specific, prior_mu, prior_logvar)+\
                klLoss(mu_shared, logvar_shared)
            preserve_loss = isometric_loss(torch.cat([mu_shared, mu_specific],dim=-1),mu_shared,m)  
            # hsic = 1000 * HSICloss(z_shared,m)
            total_loss = recon_loss + self.beta*kl_z + self.gamma * preserve_loss #+ hsic
            loss_dict = {'recon_loss':recon_loss.item(),'kl_z':kl_z.item(),
                        'preserve_loss': preserve_loss.item(),
                        # 'hsic': hsic.item(),
                        'total_loss':total_loss.item()
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

    
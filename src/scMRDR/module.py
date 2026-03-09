import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim
import torch.utils.data as Data
import numpy as np
import pandas as pd
import warnings
from .data import CombinedDataset
from .model import EmbeddingNet, EmbeddingNetXAttn
from .train import train_model, inference_model
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
import anndata as ad
from torch.utils.tensorboard import SummaryWriter
from scipy.sparse import lil_matrix,csr_matrix,issparse
import scanpy as sc
from sklearn.model_selection import train_test_split
import ot
from sklearn.neighbors import NearestNeighbors

def to_dense_array(x):
    """
    Convert input to a dense numpy array.
    Args:
        x: Input data, can be a sparse matrix, numpy array, or other types.
    Returns:
        Dense numpy array.
    """
    if issparse(x):
        return x.toarray()
    elif isinstance(x, np.ndarray):
        return x.copy()
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

class Integration:
    '''
    Integration class.
    Args:
        data: AnnData object
        layer: str, layer name in adata.layers containing the data to be integrated
        modality_key: str, key in adata.obs for modality information
        batch_key: str, key in adata.obs for batch information
        distribution: str, distribution of the data, can be "ZINB", "NB", "Normal", "Normal_positive"
        feature_list: distionary, containing unmasked feature indices for each mask group (by default, modality). Default is None, indicating all features are unmasked.
        mask_key: str, key in adata.obs to indicate mask information, corresponding to feature_list. Default is None, indicating modality_key will be used.
    '''
    def __init__(self, data, layer=None, modality_key="modality", batch_key=None, celltype_key=None, 
                 distribution = "ZINB", mask_key=None, feature_list=None):
        super(Integration,self).__init__()
        if isinstance(data, list) & isinstance(data[0], ad.AnnData):
            self.adata = ad.concat(data, axis='obs', join='inner', label="modality")
        elif isinstance(data, ad.AnnData):
            self.adata = data
        else:
            raise ValueError("Wrong type of data!")
        if layer is None:
            self.data = to_dense_array(self.adata.X)
        else:
            self.data = to_dense_array(self.adata.layers[layer])
        label_encoder = LabelEncoder()
        onehot_encoder = OneHotEncoder(sparse_output=False)
        self.modality_label = self.adata.obs[modality_key].to_numpy()
        self.modality = label_encoder.fit_transform(self.modality_label)
        self.modality_ordered = [label for label in label_encoder.classes_]
        self.modality = onehot_encoder.fit_transform(self.modality.reshape(-1, 1))

        if celltype_key is None:
            self.celltype = None
            self.celltype_ordered = None
        else:
            self.celltype_label = self.adata.obs[celltype_key].to_numpy()
            self.celltype = label_encoder.fit_transform(self.celltype_label)
            self.celltype_ordered = [label for label in label_encoder.classes_]
            self.celltype = onehot_encoder.fit_transform(self.celltype.reshape(-1, 1))    
         
        if batch_key is None:
            self.covariates = None
            self.covariates_ordered = None
        else:
            self.covariates_label = self.adata.obs[batch_key].to_numpy()
            self.covariates = label_encoder.fit_transform(self.covariates_label)
            self.covariates_ordered = [label for label in label_encoder.classes_]
            self.covariates = onehot_encoder.fit_transform(self.covariates.reshape(-1, 1))
        
        self.modality_num = self.modality.shape[1]
        
        if self.celltype is not None:
            self.celltype_num = self.celltype.shape[1]
        else:
            self.celltype_num = 0
        
        if self.covariates is not None:
            self.covariates_dim = self.covariates.shape[1]
        else:
            self.covariates_dim = 0
        
        if mask_key is None:
            self.mask = self.modality
            self.mask_num = self.modality_num
            self.mask_ordered = self.modality_ordered
        else:
            self.mask_label = self.adata.obs[mask_key].to_numpy()
            self.mask = label_encoder.fit_transform(self.mask_label)
            self.mask_ordered = [label for label in label_encoder.classes_]
            self.mask = onehot_encoder.fit_transform(self.mask.reshape(-1, 1))
            self.mask_num = self.mask.shape[1]

        if feature_list is not None:
            self.feat_mask = 0 * torch.ones(self.mask_num, self.data.shape[1])
            feature_list_ordered = [feature_list[label] for label in self.mask_ordered]
            for i, feat_idx in enumerate(feature_list_ordered):
                self.feat_mask[i, feat_idx] = 1
        else:
            self.feat_mask = torch.ones(self.mask_num, self.data.shape[1])
        
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
        else:
            raise ValueError("Distribution not recognized!")

    def setup(self, hidden_layers = [100,50], latent_dim_shared = 15, latent_dim_specific = 15, dropout_rate=0.5,
              beta = 2, gamma = 1, lambda_adv = 0.01, device=None,
              encoder_covariates=False,
              confidence_weighted=False,
              gate_mode="robust_adv",
              cw_queue_size=4096, cw_alpha=0.5, cw_c_tau=1.0,
              cw_tau_range=(0.01, 2.0), cw_tau_fallback=0.5,
              cw_eta=0.9, cw_rho=0.5, cw_tau_w=0.1, cw_w_min=0.1,
              cw_min_count=8,
              lambda_adv_base_ratio=0.35, rho_target=0.65,
              w_orphan_min=0.45, rarity_boost=0.10,
              orphan_sim_threshold=0.15, orphan_margin_threshold=0.02,
              linked_features=None,
              latent_backend="vae",
              lambda_prior_diff=1.0,
              diffusion_steps=200,
              diffusion_hidden_dim=512,
              diffusion_time_embed_dim=64,
              diffusion_beta_schedule="linear",
              diffusion_prior_cond="none",
              beta_specific=None,
              lambda_diff=None,
              diffusion_cond=None,
              model_architecture="standard",
              num_shared_tokens=4,
              num_private_tokens=2,
              token_dim=32,
              num_shared_protos=2,
              num_private_protos=2,
              xattn_depth=2,
              xattn_heads=4,
              xattn_dim_head=32,
              diff_hidden_dim=256,
              diff_steps=100,
              lambda_diff_recon=1.0,
              lambda_token_orth=0.0,
              lambda_private_cls=0.03,
              adv_stop_frac=0.25,
              support_ema_eta=0.9,
              support_threshold=0.15,
              support_min_updates=5):
        '''
        Setup the model.
        Args:
            hidden_layers: list, hidden layers dimensions of the model
            latent_dim_shared: int, latent dimension of the shared latent space
            latent_dim_specific: int, latent dimension of the specific latent space
            dropout_rate: float, dropout rate in neural network
            beta: float, beta parameter for the beta distribution
            gamma: float, gamma parameter for the gamma distribution
            lambda_adv: float, lambda parameter for the adversarial loss
            device: device to train the model. Default is None, indicating GPU will be used if available.
            confidence_weighted: bool, whether to use confidence-weighted adversarial training
            gate_mode: confidence gating backend, "legacy" or "robust_adv"
            cw_queue_size: int, per-modality FIFO queue capacity
            cw_alpha: float, fusion weight s = alpha*s_H + (1-alpha)*s_nn
            cw_c_tau: float, adaptive tau_nn multiplier
            cw_tau_range: tuple, (tau_min, tau_max) clipping range for tau_nn
            cw_tau_fallback: float, EMA fallback tau value
            cw_eta: float, tau_nn EMA decay coefficient
            cw_rho: float, budget quantile ratio
            cw_tau_w: float, gating sigmoid temperature
            cw_w_min: float, minimum weight floor
            cw_min_count: int, minimum per-modality sample count for threshold
            lambda_adv_base_ratio: ratio for base adversarial branch in robust gate mode
            rho_target: target keep ratio used by robust gate budget control
            w_orphan_min: minimum gate weight for orphan samples
            rarity_boost: rare-modality boost for robust gate
            orphan_sim_threshold: orphan threshold for cross-modal top1 similarity
            orphan_margin_threshold: orphan threshold for top1-top2 margin
            linked_features: optional list/array of linked feature indices or names
                used for raw-space MNN pairing in anchor loss
            latent_backend: "vae" or "diffusion" for shared latent modeling
            lambda_prior_diff: weight for diffusion prior loss (diffusion backend only)
            diffusion_steps: DDPM timesteps
            diffusion_hidden_dim: hidden dim for latent denoiser MLP
            diffusion_time_embed_dim: sinusoidal time embedding size
            diffusion_beta_schedule: "linear" or "cosine"
            diffusion_prior_cond: conditioning input for diffusion prior denoiser
            beta_specific: KL weight for modality-specific latent branch
        '''
        self.input_dim = self.data.shape[1]
        self.hidden_layers = hidden_layers
        self.latent_dim_shared = latent_dim_shared
        self.latent_dim_specific = latent_dim_specific
        self.dropout_rate = dropout_rate
        self.beta = beta
        self.gamma = gamma
        self.lambda_adv = lambda_adv
        if lambda_diff is not None:
            warnings.warn(
                "lambda_diff is deprecated; use lambda_prior_diff.",
                DeprecationWarning,
                stacklevel=2,
            )
            lambda_prior_diff = lambda_diff
        if diffusion_cond is not None:
            warnings.warn(
                "diffusion_cond is deprecated; use diffusion_prior_cond.",
                DeprecationWarning,
                stacklevel=2,
            )
            diffusion_prior_cond = diffusion_cond
        
        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
        else:
            self.device = device
        print("using "+str(self.device))
        self.model_architecture = model_architecture
        if self.model_architecture == "xattn" and self.distribution not in {"ZINB", "NB"}:
            raise ValueError(
                "model_architecture='xattn' in V1 requires a count-style distribution "
                "compatible with log1p inputs: {'ZINB', 'NB'}."
            )
        if self.model_architecture == "standard":
            self.model = EmbeddingNet(
                self.device, self.input_dim, self.modality_num, self.covariates_dim,
                celltype_num=self.celltype_num, layer_dims=self.hidden_layers,
                latent_dim_shared=self.latent_dim_shared, latent_dim_specific=self.latent_dim_specific,
                dropout_rate=self.dropout_rate, beta=self.beta, gamma=self.gamma, lambda_adv=self.lambda_adv,
                feat_mask=self.feat_mask, distribution=self.distribution,
                encoder_covariates=encoder_covariates,
                latent_backend=latent_backend,
                lambda_prior_diff=lambda_prior_diff,
                diffusion_steps=diffusion_steps,
                diffusion_hidden_dim=diffusion_hidden_dim,
                diffusion_time_embed_dim=diffusion_time_embed_dim,
                diffusion_beta_schedule=diffusion_beta_schedule,
                diffusion_prior_cond=diffusion_prior_cond,
                beta_specific=beta_specific,
            ).to(self.device)
        elif self.model_architecture == "xattn":
            self.model = EmbeddingNetXAttn(
                self.device, self.input_dim, self.modality_num, self.covariates_dim,
                celltype_num=self.celltype_num, backbone_dims=self.hidden_layers,
                token_dim=token_dim, num_shared_tokens=num_shared_tokens,
                num_private_tokens=num_private_tokens, dropout_rate=self.dropout_rate,
                encoder_covariates=encoder_covariates,
                beta=self.beta, beta_specific=beta_specific, lambda_adv=self.lambda_adv,
                lambda_diff_recon=lambda_diff_recon,
                lambda_token_orth=lambda_token_orth,
                lambda_private_cls=lambda_private_cls,
                diff_hidden_dim=diff_hidden_dim, diff_steps=diff_steps,
                diff_time_embed_dim=diffusion_time_embed_dim,
                diff_beta_schedule=diffusion_beta_schedule,
                xattn_depth=xattn_depth, xattn_heads=xattn_heads,
                xattn_dim_head=xattn_dim_head,
                num_shared_protos=num_shared_protos,
                num_private_protos=num_private_protos,
                feat_mask=self.feat_mask,
                distribution=self.distribution,
            ).to(self.device)
        else:
            raise ValueError("model_architecture must be 'standard' or 'xattn'")
        self.train_dataset = CombinedDataset(self.data,self.covariates,self.modality,self.mask, self.celltype)

        self.confidence_weighted = confidence_weighted
        self.cw_params = dict(
            gate_mode=gate_mode,
            cw_queue_size=cw_queue_size, cw_alpha=cw_alpha, cw_c_tau=cw_c_tau,
            cw_tau_min=cw_tau_range[0], cw_tau_max=cw_tau_range[1],
            cw_tau_fallback=cw_tau_fallback, cw_eta=cw_eta,
            cw_rho=cw_rho, cw_tau_w=cw_tau_w, cw_w_min=cw_w_min,
            cw_min_count=cw_min_count,
            lambda_adv_base_ratio=lambda_adv_base_ratio,
            rho_target=rho_target,
            w_orphan_min=w_orphan_min,
            rarity_boost=rarity_boost,
            orphan_sim_threshold=orphan_sim_threshold,
            orphan_margin_threshold=orphan_margin_threshold,
        )
        self.xattn_train_params = dict(
            adv_stop_frac=adv_stop_frac,
            support_ema_eta=support_ema_eta,
            support_threshold=support_threshold,
            support_min_updates=support_min_updates,
        )
        self._linked_features_spec = linked_features
        self.linked_feature_idx = None

    def _resolve_linked_feature_idx(self):
        """Resolve linked features lazily so non-anchor runs stay quiet."""
        if self.linked_feature_idx is not None:
            return self.linked_feature_idx

        linked_features = self._linked_features_spec
        if linked_features is not None:
            if isinstance(linked_features, torch.Tensor):
                linked_list = linked_features.detach().cpu().tolist()
            elif isinstance(linked_features, (list, tuple, np.ndarray, pd.Index, set)):
                linked_list = list(linked_features)
            else:
                raise TypeError(
                    "linked_features must be indices or feature names, got "
                    f"{type(linked_features)}"
                )

            if len(linked_list) == 0:
                linked_idx = torch.tensor([], dtype=torch.long)
            elif isinstance(linked_list[0], str):
                var_names = self.adata.var_names.astype(str)
                name_to_idx = {name: idx for idx, name in enumerate(var_names)}
                mapped_idx = [name_to_idx[name] for name in linked_list if name in name_to_idx]
                missing = len(linked_list) - len(mapped_idx)
                if missing > 0:
                    print(
                        f"Warning: {missing} linked feature names were not found in adata.var_names "
                        "and will be ignored."
                    )
                linked_idx = torch.tensor(sorted(set(mapped_idx)), dtype=torch.long)
            else:
                linked_arr = np.asarray(linked_list, dtype=np.int64)
                valid = (linked_arr >= 0) & (linked_arr < self.data.shape[1])
                if np.any(~valid):
                    dropped = int((~valid).sum())
                    print(
                        f"Warning: {dropped} linked feature indices are out of range "
                        "and will be ignored."
                    )
                linked_idx = torch.tensor(np.unique(linked_arr[valid]), dtype=torch.long)
        else:
            linked_mask = self.feat_mask.prod(dim=0)
            linked_idx = torch.where(linked_mask > 0)[0]

        self.linked_feature_idx = linked_idx
        return self.linked_feature_idx

    def train(self,epoch_num = 200, batch_size = 64, lr = 1e-5, accumulation_steps = 1,
              adaptlr = False, valid_prop = 0.1, num_warmup = 0, early_stopping = True, patience = 10,
              weighted = False,
              tensorboard = False, savepath = "./", random_state=42,
              cw_adv_ramp_epochs=10, cw_lambda_target=None,
              gate_start_epoch=None, gate_ramp_epochs=10,
              lambda_anchor=0.0, k_mnn=30,
              anchor_space="latent", anchor_start_epoch=0,
              anchor_ramp_epochs=0, anchor_sim_threshold=0.0,
              anchor_margin=0.0):
        '''
        Train the model.
        Args:
            epoch_num: int, number of epochs
            batch_size: int, batch size
            lr: float, learning rate
            accumulation_steps: int, number of steps to accumulate gradients
            adaptlr: bool, whether to adapt learning rate
            valid_prop: float, proportion of data to use for validation
            num_warmup: int, number of warmup epochs
            early_stopping: bool, whether to use early stopping
            patience: int, patience for early stopping
            weighted: bool, whether to use weighted sampling based on modality sizes
            tensorboard: bool, whether to use tensorboard
            savepath: str, path to save the tensorboard logs
            random_state: int, random seed
            cw_adv_ramp_epochs: int, number of epochs for lambda_adv to ramp from 0 to target
            cw_lambda_target: float, target lambda_adv value (None uses model's lambda_adv)
            gate_start_epoch: int or None, epoch to enable gated branch in robust mode
            gate_ramp_epochs: int, epochs to ramp gated branch weight to target
        '''
        if tensorboard:
            print("Using tensorboard!")
            self.writer = SummaryWriter(savepath)
        else:
            self.writer = None
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.lr = lr
        self.accumulation_steps = accumulation_steps
        self.adaptlr = adaptlr
        if valid_prop > 0:
            train_indices, valid_indices = train_test_split(
                np.arange(len(self.train_dataset)),
                test_size=valid_prop,
                stratify=self.modality.argmax(-1),
                random_state=random_state
            )
            train_dataset = Data.Subset(self.train_dataset, train_indices)
            valid_dataset = Data.Subset(self.train_dataset, valid_indices)
        else:
            train_dataset, valid_dataset = self.train_dataset, self.train_dataset
            train_indices = np.arange(len(train_dataset))
        self.num_batch = len(train_dataset)//self.batch_size
        
        print("Training start!")
        print(f"Model architecture: {self.model_architecture}")
        cw_kwargs = dict(
            confidence_weighted=self.confidence_weighted,
            cw_adv_ramp_epochs=cw_adv_ramp_epochs,
            cw_lambda_target=cw_lambda_target,
            gate_start_epoch=gate_start_epoch,
            gate_ramp_epochs=gate_ramp_epochs,
            **self.cw_params,
        )
        xattn_kwargs = dict(**self.xattn_train_params)
        linked_feature_idx = torch.tensor([], dtype=torch.long)
        if lambda_anchor > 0 and anchor_space != "latent":
            linked_feature_idx = self._resolve_linked_feature_idx()
            if len(linked_feature_idx) > 0:
                print(f"Linked features for MNN anchor: {len(linked_feature_idx)} features")
            else:
                print("Warning: No linked features found. Anchor loss will be disabled.")
        anchor_kwargs = dict(
            lambda_anchor=lambda_anchor if (anchor_space == "latent" or len(linked_feature_idx) > 0) else 0.0,
            k_mnn=k_mnn,
            linked_feature_idx=linked_feature_idx.to(self.device) if len(linked_feature_idx) > 0 else None,
            anchor_space=anchor_space,
            anchor_start_epoch=anchor_start_epoch,
            anchor_ramp_epochs=anchor_ramp_epochs,
            anchor_sim_threshold=anchor_sim_threshold,
            anchor_margin=anchor_margin,
        )
        if lambda_anchor > 0:
            print(f"Anchor config: space={anchor_space}, start_epoch={anchor_start_epoch}, "
                  f"ramp_epochs={anchor_ramp_epochs}, lambda={lambda_anchor}")
            print(f"  Confidence filters: sim_threshold={anchor_sim_threshold}, margin={anchor_margin}")
        if self.confidence_weighted:
            print(
                "Gate-adv config: "
                f"mode={self.cw_params['gate_mode']}, "
                f"start_epoch={gate_start_epoch}, ramp_epochs={gate_ramp_epochs}, "
                f"base_ratio={self.cw_params['lambda_adv_base_ratio']:.3f}, "
                f"rho_target={self.cw_params['rho_target']:.3f}"
            )
        if self.model_architecture == "xattn":
            print(
                "XAttn support config: "
                f"adv_stop_frac={self.xattn_train_params['adv_stop_frac']:.3f}, "
                f"support_threshold={self.xattn_train_params['support_threshold']:.3f}, "
                f"support_min_updates={self.xattn_train_params['support_min_updates']}"
            )
        if weighted:
            weights = 1.0 / np.bincount(self.modality.argmax(-1))
            sample_weights = weights[self.modality.argmax(-1)]
            sample_weights = sample_weights[train_indices]
            train_model(self.device, self.writer, train_dataset, valid_dataset,
                        self.model, self.epoch_num, self.batch_size,
                        self.num_batch, self.lr, accumulation_steps=self.accumulation_steps,
                        adaptlr=self.adaptlr, num_warmup=num_warmup, early_stopping=early_stopping,
                        patience=patience, sample_weights=sample_weights,
                        **cw_kwargs, **xattn_kwargs, **anchor_kwargs)
        else:
            train_model(self.device, self.writer, train_dataset, valid_dataset,
                        self.model, self.epoch_num, self.batch_size,
                        self.num_batch, self.lr, accumulation_steps = self.accumulation_steps,
                        adaptlr = self.adaptlr, num_warmup = num_warmup, early_stopping = early_stopping,
                        patience = patience,
                        **cw_kwargs, **xattn_kwargs, **anchor_kwargs)
        if tensorboard:
            self.writer.close()
        print("Training finished!")
    
    def inference(self, n_samples=1, dataset=None, batch_size=None, update=True, returns=False):
        '''
        Inference the model.
        Args:
            n_samples: int, number of samples to average in reparametrization trick
            dataset: dataset to use for inference
            batch_size: int, batch size
            update: bool, whether to update the latent embeddings in the adata
            returns: bool, whether to return the results, including latent shared, latent specific
        '''
        if dataset is None:
            dataset = self.train_dataset
        if batch_size is None:
            batch_size = self.batch_size
        if n_samples > 1:
            z_shared,z_specific = \
                zip(*[inference_model(self.device, dataset, self.model, batch_size) for _ in range(n_samples)])
            self.z_shared = np.mean(np.stack(z_shared, axis=0), axis=0)
            self.z_specific = np.mean(np.stack(z_specific, axis=0), axis=0) 
            # self.rho = np.mean(np.stack(rho, axis=0), axis=0) 
            # self.dispersion = np.mean(np.stack(dispersion, axis=0), axis=0) 
            # self.pi = np.mean(np.stack(pi, axis=0), axis=0) 
            # self.library_size = np.mean(np.stack(library_size, axis=0), axis=0) 
        else:
            self.z_shared,self.z_specific = \
                inference_model(self.device, dataset, self.model, batch_size)
        if update:
            self.adata.obsm['latent_shared'] = self.z_shared
            self.adata.obsm['latent_specific'] = self.z_specific
            # self.adata.layers['estimated_mean_expression'] = self.rho
            # self.adata.layers['estimated_dropout_rate'] = self.pi
            # self.adata.var['estimated_dispersion_factor'] = self.dispersion
            # self.adata.obs['estimated_library_size'] = self.library_size
            print('All results recorded in adata.')
        if returns:
            return self.z_shared,self.z_specific #,self.rho,self.dispersion,self.pi,self.library_size

    def predict(self,predict_modality,batch_size=None,strategy="observed",library_size=None,method="ot",k=10): # dataset=None,inference=False,
        '''
        Predict the missing modality data.
        Args:
            predict_modality: str, modality to predict
            batch_size: int, batch size
            strategy: str, strategy to predict the missing modality. Options (default: "observed"):
                - "observed": use the observed data from other modalities to predict the missing modality.
                - "latent": use the latent embeddings to predict the missing modality.
            library_size: array, library size for generation, default is None, indicating using the estimated library size from the model
            method: str, method to use for prediction, can be "ot" or "knn"
            k: int, number of neighbors for knn method
        Returns:
            x_pred: predicted data for the missing modality
        '''
        # if dataset is None:
        #     dataset = self.train_dataset
        if batch_size is None:
            batch_size = self.batch_size
        # if inference:
        #     z_shared,z_specific = self.inference(n_samples=1, dataset=dataset, batch_size=batch_size, update=False, returns=True)
        # else:
        #     z_shared,z_specific = self.z_shared,self.z_specific
        z_shared,z_specific = self.z_shared,self.z_specific
        curr_index = self.modality_label == predict_modality # index of measurements with the specified modality
        impt_index = self.modality_label != predict_modality
        z_shared_curr = z_shared[curr_index,:]
        z_specific_curr = z_specific[curr_index,:]
        z_shared_impt = z_shared[impt_index,:]
        # z_specific_impt = z_specific[impt_index,:]

        # z_concat_curr = np.concatenate((z_shared_curr,z_specific_curr), axis=1)
        
        if strategy == "observed":
            x_curr = self.data[curr_index,:]

            if method == "ot":
                # coupling matrix
                a = ot.unif(z_shared_impt.shape[0])
                b = ot.unif(z_shared_curr.shape[0])
                M = ot.dist(z_shared_impt, z_shared_curr, metric='euclidean')
                # W = ot.sinkhorn(a, b, M, reg=0.01)
                W = ot.emd(a, b, M)
                W = W/W.sum(axis=1,keepdims=True)
                x_pred = np.dot(W,x_curr)
            elif method == "knn":
                nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(z_shared_curr)
                distances, indices = nbrs.kneighbors(z_shared_impt)
                # weighted average
                weights = 1 / (distances + 1e-5)
                weights = weights / np.sum(weights, axis=1, keepdims=True)
                x_pred = np.array([np.sum(x_curr[indices[i]] * weights[i][:, np.newaxis], axis=0) for i in range(indices.shape[0])])
            else:
                raise ValueError("Unknown method!")  
        elif strategy == "latent":
            z_concat_curr = np.concatenate((z_shared_curr,z_specific_curr), axis=1)
            # z_specific_impt = z_specific[impt_index,:]
            # z_specific_curr_mean = np.tile(np.mean(z_specific_curr, axis=0, keepdims=True), (z_shared_impt.shape[0], 1))
            # z_concat = np.concatenate((z_shared_impt, z_specific_curr_mean), axis=1)
            if method == "ot":
                # coupling matrix
                a = ot.unif(z_shared_impt.shape[0])
                b = ot.unif(z_shared_curr.shape[0])
                M = ot.dist(z_shared_impt, z_shared_curr, metric='euclidean')
                # W = ot.sinkhorn(a, b, M, reg=0.01)
                W = ot.emd(a, b, M)
                W = W/W.sum(axis=1,keepdims=True)
                # z_specific_pred = np.dot(W, z_specific_curr)
                # z_concat = np.concatenate((z_shared_impt, z_specific_pred), axis=1)
                z_concat = np.dot(W, z_concat_curr)
            # elif method == "mean":
            #     z_specific_curr_mean = np.tile(np.mean(z_specific_curr, axis=0, keepdims=True), (z_shared_impt.shape[0], 1))
            #     z_concat = np.concatenate((z_shared_impt, z_specific_curr_mean), axis=1)
            elif method == "knn":
                nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(z_shared_curr)
                distances, indices = nbrs.kneighbors(z_shared_impt)
                # weighted average
                weights = 1 / (distances + 1e-5)
                weights = weights / np.sum(weights, axis=1, keepdims=True)
                # x_pred = np.array([np.sum(x_curr[indices[i]] * weights[i][:, np.newaxis], axis=0) for i in range(indices.shape[0])])
                # z_specific_pred = np.array([np.sum(z_specific_curr[indices[i]] * weights[i][:, np.newaxis], axis=0) for i in range(indices.shape[0])])
                # # simple average
                # z_specific_pred = np.array([np.mean(z_specific_curr[indices[i]], axis=0) for i in range(indices.shape[0])])
                z_concat = np.array([np.sum(z_concat_curr[indices[i]] * weights[i][:, np.newaxis], axis=0) for i in range(indices.shape[0])])               
            #     z_concat = np.concatenate((z_shared_impt, z_specific_pred), axis=1)

            else:
                raise ValueError("Unknown method!")    
            if self.covariates is not None:
                covariates = self.covariates[impt_index,:]
            else:
                covariates = None
            modality = np.tile(self.modality[curr_index,:][0,:], (z_concat.shape[0], 1))
            x_pred = self.generate_from_latent(z_concat,
                                                modality,
                                                covariates=covariates,
                                                library_size=library_size,
                                                n_samples=1)
        else:
            raise ValueError("Unknown strategy!")  
        return x_pred  
    
    def get_adata(self):
        '''
        Get the AnnData object with latent embeddings.
        Returns:
            AnnData object with latent embeddings in obsm.
        '''
        return self.adata
    
        
        

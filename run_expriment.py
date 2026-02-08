import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scanpy as sc
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast

# Ensure scmrdr modules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "src"))

from scmrdr.model import ZINBDiffusion, Denoise_net
from scmrdr.utils import gaussian_parameters, mean_flat

# ==========================================
# 1. Custom Causal Encoder
# ==========================================

class SimpleCausalEncoder(nn.Module):
    """
    A simplified VAE-style encoder that supports Causal DAG constraints.
    If an adjacency matrix (DAG) is provided, it enforces structural equation constraints
    on the latent space. Otherwise, it acts as a standard Gaussian encoder.
    """
    def __init__(self, input_dim, num_factors, hidden_dims=[512, 256], adj_matrix=None):
        super().__init__()
        self.input_dim = input_dim
        self.num_factors = num_factors
        
        # Build MLP backbone
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU())
            in_dim = h_dim
        self.backbone = nn.Sequential(*layers)
        
        # Output layers for Mean and LogVar
        # We produce embeddings for 'num_factors'
        self.fc_mean = nn.Linear(in_dim, num_factors)
        self.fc_logvar = nn.Linear(in_dim, num_factors)
        
        # Causal Logic
        if adj_matrix is not None:
            print("[Model] Initializing with Causal Graph constraints.")
            # Ensure adj_matrix is a Tensor
            if not isinstance(adj_matrix, torch.Tensor):
                adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
            
            # Register DAG as a buffer (fixed structure) or parameter (learnable)
            # Here we assume fixed structure provided by user
            self.register_buffer('causal_dag', adj_matrix)
            self.register_buffer('I', torch.eye(num_factors))
            self.use_causal = True
        else:
            print("[Model] No Causal Graph provided. Initializing standard Encoder.")
            self.use_causal = False
            self.register_buffer('causal_dag', None)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        # 1. Encode to exogenous variables (u)
        h = self.backbone(x)
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        
        # Sample exogenous factors u
        u = self.reparameterize(mean, logvar)
        
        dag_loss = torch.tensor(0.0, device=x.device)
        z = u

        # 2. Apply Causal Structural Equations if enabled
        # Model: z = A z + u  =>  (I - A) z = u  =>  z = (I - A)^-1 u
        if self.use_causal:
            # Check dimensions
            if self.causal_dag.shape[0] != self.num_factors:
                # Fallback if dimensions mismatch (robustness)
                pass 
            else:
                # Calculate z from u
                # z = (I - A)^-1 @ u
                # Note: u is (batch, num_factors), we need to transpose for matmul
                inv_mat = torch.inverse(self.I - self.causal_dag)
                z = torch.matmul(u, inv_mat.t()) # (batch, num_factors)
                
                # Calculate consistency loss (DAG constraint)
                # We want to ensure the generated z respects z = Az + u
                # Reconstruct u from z: u_recon = z - z @ A.t()
                u_recon = z - torch.matmul(z, self.causal_dag.t())
                dag_loss = F.mse_loss(u_recon, u)

        # 3. KL Divergence (Standard Normal Prior on u)
        # KL(N(mean, var) || N(0, 1))
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()

        return z, kl_loss, dag_loss


class CausalDiffusionModel(nn.Module):
    """
    Wrapper model combining the Causal Encoder and the ZINB Diffusion Decoder.
    """
    def __init__(self, input_dim, num_factors, adj_matrix=None, hidden_dim=512):
        super().__init__()
        
        # Encoder
        self.encoder = SimpleCausalEncoder(
            input_dim=input_dim, 
            num_factors=num_factors, 
            adj_matrix=adj_matrix
        )
        
        # Diffusion Decoder
        # The context dimension for diffusion is the latent size (num_factors)
        self.denoise_net = Denoise_net(
            dim=input_dim, 
            out_dim=input_dim, 
            hidden_dim=hidden_dim,
            context_dim=num_factors, # Condition on latent z
            dim_head=64,
            num_heads=4,
            depth=2
        )
        
        self.diffusion = ZINBDiffusion(
            denoise_fn=self.denoise_net,
            profile_size=input_dim,
            timesteps=1000,
            loss_type="l2"
        )

    def forward(self, x):
        # Preprocessing (Log1p is standard for scRNA-seq in this pipeline)
        x_log = torch.log1p(x)
        
        # 1. Encode
        z, kl_loss, dag_loss = self.encoder(x_log)
        
        # 2. Diffusion Loss
        # We condition the diffusion process on the latent vector z
        # z needs to be shaped for attention: (batch, 1, num_factors) or similar
        z_context = z.unsqueeze(1) 
        
        # ZINBDiffusion calculates the loss internally
        diff_loss = self.diffusion(x_log, embeddings=z_context)
        
        return diff_loss, kl_loss, dag_loss

    def sample(self, num_samples, device):
        """Generate samples from the model."""
        # Sample from prior N(0,1) for u, then transform to z
        u = torch.randn(num_samples, self.encoder.num_factors).to(device)
        
        if self.encoder.use_causal:
            inv_mat = torch.inverse(self.encoder.I - self.encoder.causal_dag)
            z = torch.matmul(u, inv_mat.t())
        else:
            z = u
            
        z_context = z.unsqueeze(1)
        
        # Sample from Diffusion
        samples = self.diffusion.sample_with_factor(z_context, batch_size=num_samples)
        return samples

# ==========================================
# 2. Helper Functions
# ==========================================

def load_data(data_path, adj_path=None, num_factors=20):
    """
    Robust data loading function.
    """
    print(f"[Data] Loading expression matrix from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
        
    adata = sc.read_h5ad(data_path)
    
    # Use counts if available, else X
    if 'counts' in adata.layers:
        X = adata.layers['counts']
    else:
        X = adata.X
        
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    adj_matrix = None
    if adj_path and os.path.exists(adj_path):
        print(f"[Data] Loading adjacency matrix from {adj_path}...")
        try:
            adj_df = pd.read_csv(adj_path, index_col=0)
            adj_matrix = adj_df.values
            
            # Check dimensions
            if adj_matrix.shape != (num_factors, num_factors):
                print(f"[Warning] Adjacency matrix shape {adj_matrix.shape} does not match num_factors ({num_factors}).")
                print("[Warning] Ignoring adjacency matrix to prevent errors.")
                adj_matrix = None
            else:
                print("[Data] Adjacency matrix loaded successfully.")
        except Exception as e:
            print(f"[Error] Failed to load adjacency matrix: {e}")
            adj_matrix = None
    else:
        if adj_path:
            print(f"[Warning] Adjacency file not found: {adj_path}")
        print("[Data] Proceeding without Causal Graph.")
        
    return X, adj_matrix

# ==========================================
# 3. Main Training Script
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Train Causal Diffusion Model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to .h5ad file")
    parser.add_argument("--adj_path", type=str, default=None, help="Path to adjacency matrix .csv (optional)")
    parser.add_argument("--save_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--num_factors", type=int, default=20, help="Dimension of latent space (and DAG nodes)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--kl_weight", type=float, default=0.001, help="Weight for KL divergence loss")
    parser.add_argument("--dag_weight", type=float, default=1.0, help="Weight for DAG consistency loss")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps to save memory")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension for Denoise Net")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[System] Running on device: {device}")
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    # Load Data
    X, adj_matrix = load_data(args.data_path, args.adj_path, args.num_factors)
    input_dim = X.shape[1]
    print(f"[Data] Input dimension (genes): {input_dim}")
    
    # Prepare DataLoader
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize Model
    model = CausalDiffusionModel(
        input_dim=input_dim,
        num_factors=args.num_factors,
        adj_matrix=adj_matrix,
        hidden_dim=args.hidden_dim
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler() # For Mixed Precision Training
    
    # Training Loop
    loss_history = []
    print("[Train] Starting training...")
    
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_diff = 0
        epoch_kl = 0
        epoch_dag = 0
        
        pbar = tqdm(enumerate(dataloader), desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch", total=len(dataloader))
        
        for step, batch in pbar:
            x_batch = batch[0].to(device)
            
            # Mixed Precision Context
            with autocast(enabled=True):
                diff_loss, kl_loss, dag_loss = model(x_batch)
                total_loss = diff_loss + args.kl_weight * kl_loss + args.dag_weight * dag_loss
                # Normalize loss for gradient accumulation
                total_loss = total_loss / args.grad_accum
            
            # Scale loss and backward
            scaler.scale(total_loss).backward()
            
            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            # Logging
            # Multiply back by grad_accum to log the actual loss value
            epoch_loss += total_loss.item() * args.grad_accum
            epoch_diff += diff_loss.item()
            epoch_kl += kl_loss.item()
            epoch_dag += dag_loss.item()
            
            pbar.set_postfix({
                'Loss': f"{total_loss.item():.2f}", 
                'Diff': f"{diff_loss.item():.2f}",
                'DAG': f"{dag_loss.item():.2f}"
            })
            
        # Average losses
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Total={avg_loss:.4f}, Diff={epoch_diff/len(dataloader):.4f}, KL={epoch_kl/len(dataloader):.4f}, DAG={epoch_dag/len(dataloader):.4f}")
            
    # Save Model
    save_path = os.path.join(args.save_dir, "causal_diffusion_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"[Save] Model saved to {save_path}")
    
    # Plot Loss
    plt.figure()
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(args.save_dir, "loss_curve.png"))
    print("[Save] Loss curve saved.")
    
    print("[Done] Training finished.")

if __name__ == "__main__":
    main()

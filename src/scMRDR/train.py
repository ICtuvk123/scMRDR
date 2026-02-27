import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from .anchor import find_mnn_pairs, find_mnn_pairs_latent, anchor_loss


def _log_modality_scalars(writer, prefix, values, global_step):
    for mod_idx, value in values.items():
        writer.add_scalar(f"{prefix}/m{mod_idx}", value, global_step)

class EarlyStopping:
    '''
    Early stopping for training.
    Args:
        patience: int, patience for early stopping
        delta: float, delta for early stopping
        verbose: bool, whether to print early stopping information
    '''
    def __init__(self, patience=10, delta=0.0, verbose=False):
        super().__init__()
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(model)
            self.counter = 0

    # def save_checkpoint(self, model, path="best_model.pt"):
    #     self.path = path
        # torch.save(model.state_dict(), self.path)
        # if self.verbose:
        #     print(f"Validation loss decreased, model saved to {self.path}")

def train_model(device, writer, train_dataset, validate_dataset, model, epoch_num, batch_size,
                num_batch, lr, accumulation_steps=1, num_warmup = 0, adaptlr = False, early_stopping=True, patience=25,
                sample_weights=None,
                confidence_weighted=False,
                gate_mode="robust_adv",
                cw_queue_size=4096, cw_alpha=0.5, cw_c_tau=1.0,
                cw_tau_min=0.01, cw_tau_max=2.0, cw_tau_fallback=0.5,
                cw_eta=0.9, cw_rho=0.5, cw_tau_w=0.1, cw_w_min=0.1,
                cw_min_count=8, cw_adv_ramp_epochs=10, cw_lambda_target=None,
                gate_start_epoch=None, gate_ramp_epochs=10,
                lambda_adv_base_ratio=0.35, rho_target=0.65,
                w_orphan_min=0.45, rarity_boost=0.10,
                orphan_sim_threshold=0.15, orphan_margin_threshold=0.02,
                lambda_anchor=0.0, k_mnn=30, linked_feature_idx=None,
                anchor_space="latent", anchor_start_epoch=0,
                anchor_ramp_epochs=0, anchor_sim_threshold=0.0,
                anchor_margin=0.0):
    '''
    Train the model.
    Args:
        device: device to train the model
        writer: writer to write the training progress
        train_dataset: train dataset
        validate_dataset: validate dataset
        model: model to train
        epoch_num: number of epochs
        batch_size: batch size
        num_batch: number of batches
        lr: learning rate
        accumulation_steps: number of steps to accumulate gradients
        num_warmup: number of warmup epochs
        adaptlr: whether to adapt learning rate
        early_stopping: whether to use early stopping
        patience: patience for early stopping
        sample_weights: sample weights for weighted sampling
        confidence_weighted: whether to use confidence-weighted adversarial training
        gate_mode: "legacy" or "robust_adv"
        cw_*: confidence weighting hyperparameters
    '''
    # load data
    if sample_weights is not None:
        sample_weights = torch.tensor(sample_weights,dtype=torch.double)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_data = DataLoader(train_dataset,batch_size,shuffle=False,sampler=sampler,drop_last=True,num_workers=4,pin_memory=True)
    else:   
        train_data = DataLoader(train_dataset,batch_size,shuffle=True,drop_last=True,num_workers=4,pin_memory=True)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    if hasattr(model, "vae_parameters"):
        vae_params = list(model.vae_parameters())
    else:
        vae_params = (
            list(model.encoder_shared.parameters())
            + list(model.encoder_specific.parameters())
            + list(model.decoder.parameters())
            + list(model.prior_net_specific.parameters())
        )
    optimizer_vae = torch.optim.Adam(vae_params, lr=lr)
    optimizer_d = torch.optim.Adam(model.discriminator.parameters(), lr=lr)
    if adaptlr==True:
        scheduler_d =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer_d,
                                                            T_max =  epoch_num * num_batch)
        scheduler_vae =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer_vae,
                                                            T_max =  epoch_num * num_batch)
    
    if early_stopping:
        early_stopping = EarlyStopping(patience=patience, verbose=True)

    # Confidence-weighted adversarial training setup
    cw = None
    if confidence_weighted:
        from .confidence import ConfidenceWeighter, RobustAdvGate
        if gate_mode == "legacy":
            cw = ConfidenceWeighter(
                latent_dim=model.latent_dim_shared,
                num_modalities=model.modality_num,
                device=device,
                queue_size=cw_queue_size, alpha=cw_alpha,
                c_tau=cw_c_tau, tau_min=cw_tau_min, tau_max=cw_tau_max,
                tau_fallback=cw_tau_fallback, eta=cw_eta,
                rho=cw_rho, tau_w=cw_tau_w, w_min=cw_w_min,
                min_count=cw_min_count,
            )
        elif gate_mode == "robust_adv":
            cw = RobustAdvGate(
                latent_dim=model.latent_dim_shared,
                num_modalities=model.modality_num,
                device=device,
                queue_size=cw_queue_size, alpha=cw_alpha,
                c_tau=cw_c_tau, tau_min=cw_tau_min, tau_max=cw_tau_max,
                tau_fallback=cw_tau_fallback, eta=cw_eta,
                tau_w=cw_tau_w, min_count=cw_min_count,
                rho_target=rho_target, w_floor=cw_w_min,
                w_orphan_min=w_orphan_min, rarity_boost=rarity_boost,
                orphan_sim_threshold=orphan_sim_threshold,
                orphan_margin_threshold=orphan_margin_threshold,
            )
        else:
            raise ValueError(f"Unsupported gate_mode: {gate_mode}")
        lambda_target = cw_lambda_target if cw_lambda_target is not None else model.lambda_adv
        T_w = num_warmup
        T_r = cw_adv_ramp_epochs
        if gate_start_epoch is None:
            gate_start_epoch = num_warmup

    for epoch in range(epoch_num):
        # Anchor ramp: Phase A (no anchor) -> Phase B (ramp up)
        if lambda_anchor > 0 and epoch >= anchor_start_epoch:
            if anchor_ramp_epochs > 0:
                ramp = min(1.0, (epoch - anchor_start_epoch) / anchor_ramp_epochs)
            else:
                ramp = 1.0
            lambda_anchor_current = lambda_anchor * ramp
        else:
            lambda_anchor_current = 0.0

        model.train()
        total_loss,recon_loss,kl_z,preserve_loss,adv_loss,total_discri_loss,total_anchor_loss = \
            0,0,0,0,0,0,0
        for step, (X,b,m,i,w) in enumerate(train_data):
            X,b,m,i,w = X.to(device),b.to(device),m.to(device), i.to(device),w.to(device)
            X.requires_grad = True
            b.requires_grad = True
            m.requires_grad = True
            
            # torch.autograd.set_detect_anomaly(True)
            # with torch.autograd.detect_anomaly():
            #     loss.backward() 

            if epoch < num_warmup:
                model.train()
                mu_shared, _, loss, loss_dict = model(X,b,m,i,w,stage="warmup")

                # Anchor loss (MNN pairs)
                modality_labels_batch = torch.argmax(m, dim=1)
                a_loss_val = 0.0
                if lambda_anchor_current > 0:
                    if anchor_space == "latent":
                        mnn_i, mnn_j = find_mnn_pairs_latent(
                            mu_shared, modality_labels_batch, k=k_mnn,
                            sim_threshold=anchor_sim_threshold, margin=anchor_margin,
                        )
                    else:
                        mnn_i, mnn_j = find_mnn_pairs(
                            X, modality_labels_batch, model.feat_mask, k=k_mnn,
                            linked_feature_idx=linked_feature_idx,
                            sim_threshold=anchor_sim_threshold, margin=anchor_margin,
                        )
                    a_loss = anchor_loss(mu_shared, mnn_i, mnn_j)
                    loss = loss + lambda_anchor_current * a_loss
                    a_loss_val = a_loss.item()

                # with torch.autograd.detect_anomaly():
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                if (step + 1) % accumulation_steps == 0:
                    optimizer_vae.step()
                    optimizer_vae.zero_grad()
                if (writer is not None) & (adaptlr == True):
                    writer.add_scalar("lr_vae/train",scheduler_vae.get_last_lr()[0],epoch*num_batch+step)
                if adaptlr == True:
                    scheduler_vae.step()

                # Update confidence weighter queues during warmup
                if cw is not None and '_z_shared' in loss_dict:
                    cw.update_queues(loss_dict['_z_shared'].detach(), modality_labels_batch)

                total_loss+=loss.item()
                recon_loss+=loss_dict['recon_loss']
                kl_z+=loss_dict['kl_z']
                preserve_loss+=loss_dict['preserve_loss']
                total_anchor_loss+=a_loss_val

                # print(loss)
                if writer is not None:
                    writer.add_scalar("Loss/train", loss.item(), epoch*num_batch+step+1)
                    writer.add_scalar("recon_Loss/train", loss_dict['recon_loss'], epoch*num_batch+step+1)
                    writer.add_scalar("KLz_Loss/train", loss_dict['kl_z'], epoch*num_batch+step+1)
                    writer.add_scalar("preserve_Loss/train", loss_dict['preserve_loss'], epoch*num_batch+step+1)
                    writer.add_scalar("anchor_Loss/train", a_loss_val, epoch*num_batch+step+1)

            elif epoch >= num_warmup:
                ### === Phase A: Train Discriminator === ###
                model.eval()
                model.discriminator.train()
                discri_loss = model(X,b,m,i,w, stage="discriminator")
                # with torch.autograd.detect_anomaly():
                discri_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1) 
                if (step + 1) % accumulation_steps == 0:
                    optimizer_d.step()
                    optimizer_d.zero_grad()  
                if (writer is not None) & (adaptlr == True):
                    writer.add_scalar("lr_d/train",scheduler_d.get_last_lr()[0],epoch*num_batch+step)
                if adaptlr == True:    
                    scheduler_d.step()

                ### === Phase B: Train VAE, fool Discriminator === ###
                model.train()
                model.discriminator.eval()

                if cw is not None:
                    # Confidence-weighted adversarial training
                    mu_shared, _, base_loss, loss_dict = model(X,b,m,i,w,stage="vae",return_adv_components=True)

                    z_shared_batch = loss_dict['_z_shared']
                    logits_batch = loss_dict['_modality_logits']
                    per_sample_adv = loss_dict['_per_sample_adv']
                    modality_labels_batch = torch.argmax(m, dim=1)

                    # Lambda ramp
                    ramp = min(1.0, max(0.0, (epoch - T_w) / T_r)) if T_r > 0 else 1.0
                    lambda_adv_current = lambda_target * ramp

                    if gate_mode == "robust_adv":
                        gate_info = cw.compute_weights(z_shared_batch, modality_labels_batch, logits_batch)
                        adv_weights = gate_info["weights"]
                        cw.update_queues(z_shared_batch.detach(), modality_labels_batch)

                        base_ratio = min(max(lambda_adv_base_ratio, 0.0), 1.0)
                        lambda_adv_base = lambda_adv_current * base_ratio
                        gate_lambda_full = lambda_adv_current * (1.0 - base_ratio)
                        if epoch < gate_start_epoch:
                            gate_ramp = 0.0
                        else:
                            gate_ramp = (
                                min(1.0, (epoch - gate_start_epoch) / gate_ramp_epochs)
                                if gate_ramp_epochs > 0 else 1.0
                            )
                        lambda_adv_gate = gate_lambda_full * gate_ramp

                        adv_loss_base = per_sample_adv.sum() / m.shape[0]
                        adv_loss_weighted = (adv_weights * per_sample_adv).sum() / m.shape[0]
                        adv_total = lambda_adv_base * adv_loss_base + lambda_adv_gate * adv_loss_weighted
                        loss = base_loss + adv_total
                        loss_dict_adv_val = adv_total.item()
                        adv_base_val = adv_loss_base.item()
                        adv_gate_val = adv_loss_weighted.item()
                    else:
                        adv_weights = cw.compute_weights(z_shared_batch, modality_labels_batch, logits_batch)
                        cw.update_queues(z_shared_batch.detach(), modality_labels_batch)
                        adv_loss_weighted = (adv_weights * per_sample_adv).sum() / m.shape[0]
                        loss = base_loss + lambda_adv_current * adv_loss_weighted
                        loss_dict_adv_val = (lambda_adv_current * adv_loss_weighted).item()

                    # Anchor loss (MNN pairs)
                    a_loss_val = 0.0
                    if lambda_anchor_current > 0:
                        if anchor_space == "latent":
                            mnn_i, mnn_j = find_mnn_pairs_latent(
                                mu_shared, modality_labels_batch, k=k_mnn,
                                sim_threshold=anchor_sim_threshold, margin=anchor_margin,
                            )
                        else:
                            mnn_i, mnn_j = find_mnn_pairs(
                                X, modality_labels_batch, model.feat_mask, k=k_mnn,
                                linked_feature_idx=linked_feature_idx,
                                sim_threshold=anchor_sim_threshold, margin=anchor_margin,
                            )
                        a_loss = anchor_loss(mu_shared, mnn_i, mnn_j)
                        loss = loss + lambda_anchor_current * a_loss
                        a_loss_val = a_loss.item()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    if (step + 1) % accumulation_steps == 0:
                        optimizer_vae.step()
                        optimizer_vae.zero_grad()

                    # Logging
                    total_loss += loss.item()
                    recon_loss += loss_dict['recon_loss']
                    kl_z += loss_dict['kl_z']
                    preserve_loss += loss_dict['preserve_loss']
                    adv_loss += loss_dict_adv_val
                    total_discri_loss += discri_loss.item()
                    total_anchor_loss += a_loss_val

                    if writer is not None:
                        global_step = epoch * num_batch + step + 1
                        writer.add_scalar("Loss/train", loss.item(), global_step)
                        writer.add_scalar("recon_Loss/train", loss_dict['recon_loss'], global_step)
                        writer.add_scalar("KLz_Loss/train", loss_dict['kl_z'], global_step)
                        writer.add_scalar("preserve_Loss/train", loss_dict['preserve_loss'], global_step)
                        writer.add_scalar("adv_Loss/train", loss_dict_adv_val, global_step)
                        writer.add_scalar("discri_Loss/train", discri_loss.item(), global_step)
                        writer.add_scalar("lambda_adv/train", lambda_adv_current, global_step)
                        writer.add_scalar("mean_adv_weight/train", adv_weights.mean().item(), global_step)
                        writer.add_scalar("min_adv_weight/train", adv_weights.min().item(), global_step)
                        writer.add_scalar("tau_nn/train", cw.current_tau_nn, global_step)
                        if gate_mode == "robust_adv":
                            writer.add_scalar("tau_h/train", cw.current_tau_h, global_step)
                            writer.add_scalar("adv_base_unscaled/train", adv_base_val, global_step)
                            writer.add_scalar("adv_gate_unscaled/train", adv_gate_val, global_step)
                            writer.add_scalar("lambda_adv_gate/train", lambda_adv_gate, global_step)
                            _log_modality_scalars(
                                writer,
                                "mean_adv_weight_by_modality/train",
                                cw.current_mean_weight_by_modality,
                                global_step,
                            )
                            _log_modality_scalars(
                                writer,
                                "orphan_ratio_by_modality/train",
                                cw.current_orphan_ratio_by_modality,
                                global_step,
                            )
                            _log_modality_scalars(
                                writer,
                                "gate_threshold_by_modality/train",
                                cw.current_modality_thresholds,
                                global_step,
                            )
                        writer.add_scalar("anchor_Loss/train", a_loss_val, global_step)
                else:
                    # Original adversarial training (no confidence weighting)
                    mu_shared, _, loss, loss_dict = model(X,b,m,i,w,stage="vae")

                    # Anchor loss (MNN pairs)
                    a_loss_val = 0.0
                    if lambda_anchor_current > 0:
                        modality_labels_batch = torch.argmax(m, dim=1)
                        if anchor_space == "latent":
                            mnn_i, mnn_j = find_mnn_pairs_latent(
                                mu_shared, modality_labels_batch, k=k_mnn,
                                sim_threshold=anchor_sim_threshold, margin=anchor_margin,
                            )
                        else:
                            mnn_i, mnn_j = find_mnn_pairs(
                                X, modality_labels_batch, model.feat_mask, k=k_mnn,
                                linked_feature_idx=linked_feature_idx,
                                sim_threshold=anchor_sim_threshold, margin=anchor_margin,
                            )
                        a_loss = anchor_loss(mu_shared, mnn_i, mnn_j)
                        loss = loss + lambda_anchor_current * a_loss
                        a_loss_val = a_loss.item()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    if (step + 1) % accumulation_steps == 0:
                        optimizer_vae.step()
                        optimizer_vae.zero_grad()

                    total_loss+=loss.item()
                    recon_loss+=loss_dict['recon_loss']
                    kl_z+=loss_dict['kl_z']
                    preserve_loss+=loss_dict['preserve_loss']
                    adv_loss+=loss_dict['adv_loss']
                    total_discri_loss+=discri_loss.item()
                    total_anchor_loss+=a_loss_val

                    if writer is not None:
                        writer.add_scalar("Loss/train", loss.item(), epoch*num_batch+step+1)
                        writer.add_scalar("recon_Loss/train", loss_dict['recon_loss'], epoch*num_batch+step+1)
                        writer.add_scalar("KLz_Loss/train", loss_dict['kl_z'], epoch*num_batch+step+1)
                        writer.add_scalar("preserve_Loss/train", loss_dict['preserve_loss'], epoch*num_batch+step+1)
                        writer.add_scalar("adv_Loss/train", loss_dict['adv_loss'], epoch*num_batch+step+1)
                        writer.add_scalar("discri_Loss/train", discri_loss.item(), epoch*num_batch+step+1)
                        writer.add_scalar("anchor_Loss/train", a_loss_val, epoch*num_batch+step+1)

                if (writer is not None) & (adaptlr == True):
                    writer.add_scalar("lr_vae/train",scheduler_vae.get_last_lr()[0],epoch*num_batch+step)
                if adaptlr == True:
                    scheduler_vae.step()
            
        if writer is not None:
            writer.add_scalar("Loss_epoch/train", total_loss / num_batch, epoch+1)
            writer.add_scalar("recon_Loss_epoch/train", recon_loss / num_batch, epoch+1)
            writer.add_scalar("KLz_Loss_epoch/train", kl_z / num_batch, epoch+1)
            writer.add_scalar("preserve_Loss_epoch/train", preserve_loss / num_batch, epoch+1)
            writer.add_scalar("adv_Loss_epoch/train", adv_loss / num_batch, epoch+1)
            writer.add_scalar("discri_Loss_epoch/train", total_discri_loss / num_batch, epoch+1)
            writer.add_scalar("anchor_Loss_epoch/train", total_anchor_loss / num_batch, epoch+1)

        if (epoch + 1) % 1 == 0:
            print("epoch {}: loss = {:.4f}, Recon = {:.4f}, KL = {:.4f}, preserve = {:.4f}, adv = {:.4f}, discri = {:.4f}, anchor = {:.4f}".format(
                epoch+1,total_loss / num_batch, recon_loss / num_batch, kl_z / num_batch, preserve_loss / num_batch, adv_loss / num_batch, total_discri_loss / num_batch, total_anchor_loss / num_batch))
        
        if epoch >= num_warmup:
            if early_stopping:
                validate_loss = validate_model(device, validate_dataset, model, batch_size)
                early_stopping(validate_loss, model)
                if early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

def validate_model(device, validate_dataset, model, batch_size):
    '''
    Validate the model.
    Args:
        device: device to validate the model
        validate_dataset: validate dataset
        model: model to validate
        batch_size: batch size
    '''
    model.eval()
    validate_data = DataLoader(validate_dataset,batch_size,shuffle=False,drop_last=False,num_workers=4,pin_memory=True)
    total_loss= 0
    with torch.no_grad():
        for _, (X,b,m,i,w) in enumerate(validate_data):
            X,b,m,i,w = X.to(device),b.to(device),m.to(device), i.to(device),w.to(device)
            _,_,loss,_ = model(X,b,m,i,w,stage="vae")
            total_loss += loss.item()
    return total_loss / max(len(validate_data), 1)


def inference_model(device, inference_dataset, model, batch_size):
    '''
    Inference the model.
    Args:
        device: device to inference the model
        inference_dataset: inference dataset
        model: model to inference
        batch_size: batch size
    '''
    model.eval()
    inference_data = DataLoader(inference_dataset,batch_size,shuffle=False,drop_last=False,num_workers=4,pin_memory=True)
    z1_list = []
    z2_list = []
    total_loss,recon_loss,kl_z,preserve_loss,adv_loss= \
            0,0,0,0,0
    for step, (X,b,m,i,w) in enumerate(inference_data):
        X,b,m,i,w = X.to(device),b.to(device),m.to(device), i.to(device),w.to(device)
        z1,z2,loss,loss_dict = model(X,b,m,i,w,stage="vae")
        z1_list.append(z1.detach().cpu().numpy())
        z2_list.append(z2.detach().cpu().numpy())
        total_loss+=loss.item()
        recon_loss+=loss_dict['recon_loss']
        kl_z+=loss_dict['kl_z']
        preserve_loss+=loss_dict['preserve_loss']
        adv_loss+=loss_dict['adv_loss']
        # total_discri_loss+=discri_loss.item()
        
    z_shared = np.concatenate(z1_list, axis=0)
    z_specific = np.concatenate(z2_list, axis=0)
    num_batch = np.ceil(len(inference_dataset)/batch_size)
    print("inference: loss = {:.4f}, Recon_loss = {:.4f}, KL_loss = {:.4f}, preserve_loss = {:.4f}, adv_loss = {:.4f}".format( # 
          total_loss / num_batch, recon_loss / num_batch, kl_z / num_batch, preserve_loss / num_batch, adv_loss / num_batch))  #
    
    return z_shared, z_specific

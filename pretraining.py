import os
import pickle
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from data_openml import DataSetCatCon_imputedX
from augmentations import embed_data_mask, add_noise  # mixup_data is imported dynamically if needed
from corruptor import Corruptor


def DeepIFSAC_pretrain(model, cat_idxs, X_train, y_train, X_train_imp, train_mask,
                        continuous_mean_std, imp_continuous_mean_std, opt, device):
    """
    Pretrains a given DeepIFSAC model using specified training data and options.
    
    Pretraining applies data augmentations (e.g., cutmix, mixup, denoising, contrastive)
    and computes losses over multiple epochs. Training metrics are stored in a pickle file.
    
    Parameters:
        model (torch.nn.Module): The DeepIFSAC model to pretrain.
        cat_idxs (list of int): Categorical feature indices.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_train_imp (tensor): Imputed training features (tensor).
        train_mask (tensor): Mask indicating missing data.
        continuous_mean_std (tuple): Mean and std for continuous features.
        imp_continuous_mean_std (tuple): Mean and std for imputed continuous features.
        opt (Namespace): Training hyperparameters and options.
        device (torch.device): Device to use (CPU or GPU).
        
    Returns:
        model (torch.nn.Module): The pretrained model.
        nrmse_con (float): NRMSE for continuous features.
        nrmse_cat (float): NRMSE for categorical features (set to 0 in this refactor).
    """
    # Create directory and load/save metrics
    directory = './results/training_scores'
    os.makedirs(directory, exist_ok=True)
    filename = f'{directory}/pretrain_{opt.dset_id}_{opt.attentiontype}_{opt.missing_type}__{opt.corruption_type}.pkl'
    
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            try:
                metrics_dict = pickle.load(f)
            except EOFError:
                metrics_dict = {}
    else:
        metrics_dict = {}

    missing_rate_key = f'missing_{opt.missing_rate}'
    metrics_dict.setdefault(missing_rate_key, {})
    fold_key = f'fold_{opt.dset_seed}'
    metrics_dict[missing_rate_key].setdefault(fold_key, {'epochs': {}})

    # Prepare dataset and DataLoader
    train_ds = DataSetCatCon_imputedX(X_train, X_train_imp.cpu().numpy(), y_train, train_mask,
                                      cat_idxs, opt.dtask, continuous_mean_std, imp_continuous_mean_std)
    trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True, num_workers=0)

    # Set up the corruptor for noise augmentation
    corruptor_settings = {
        'method': 'draw',
        'missing': opt.missing_rate,
        'missing_type': 'mcar',
        'mice': 'LinearRegression'
    }
    corruptor = Corruptor(X_train_imp, corruptor_settings)

    vision_dset = opt.vision_dset
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    pt_aug_dict = {'noise_type': opt.pt_aug, 'lambda': opt.pt_aug_lam}
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()

    print("Pretraining begins!")
    all_predictions, all_original_data = [], []
    all_predictions_cat, all_original_cat = [], []
    train_mask_t = []

    for epoch in range(opt.pretrain_epochs):
        running_loss, num_batches = 0.0, 0
        model.train()
        for batch in trainloader:
            optimizer.zero_grad()
            # Unpack and move batch to device
            x_categ, x_cont, x_categ_imp, x_cont_imp, _, cat_mask, con_mask, train_mask_batch = \
                [d.to(device) for d in batch]
            # Data augmentation: cutmix and mixup
            if 'cutmix' in opt.pt_aug:
                x_categ_corr, x_cont_corr = add_noise(
                    x_categ_imp, x_cont_imp,
                    noise_params=pt_aug_dict,
                    mr=opt.missing_rate, mt=opt.missing_type,
                    corruptor1=corruptor, opt=opt
                )
                _, x_categ_enc_2, x_cont_enc_2 = embed_data_mask(x_categ_corr, x_cont_corr, cat_mask, con_mask, model, vision_dset)
            else:
                _, x_categ_enc_2, x_cont_enc_2 = embed_data_mask(x_categ_imp, x_cont_imp, cat_mask, con_mask, model, vision_dset)

            _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ_imp, x_cont_imp, cat_mask, con_mask, model, vision_dset)

            loss = 0.0

            # Contrastive loss (only one block is retained)
            if 'contrastive' in opt.pt_tasks:
                aug1 = model.transformer(x_categ_enc, x_cont_enc)
                aug2 = model.transformer(x_categ_enc_2, x_cont_enc_2)
                aug1 = (aug1 / aug1.norm(dim=-1, keepdim=True)).flatten(1, 2)
                aug2 = (aug2 / aug2.norm(dim=-1, keepdim=True)).flatten(1, 2)
                if opt.pt_projhead_style == 'diff':
                    aug1 = model.pt_mlp(aug1)
                    aug2 = model.pt_mlp2(aug2)
                elif opt.pt_projhead_style == 'same':
                    aug1 = model.pt_mlp(aug1)
                    aug2 = model.pt_mlp(aug2)
                else:
                    print('Not using projection head')
                logits1 = aug1 @ aug2.t() / opt.nce_temp
                logits2 = aug2 @ aug1.t() / opt.nce_temp
                targets = torch.arange(logits1.size(0)).to(logits1.device)
                loss += opt.lam0 * (criterion1(logits1, targets) + criterion1(logits2, targets)) / 2

            elif 'contrastive_sim' in opt.pt_tasks:
                aug1 = model.transformer(x_categ_enc, x_cont_enc)
                aug2 = model.transformer(x_categ_enc_2, x_cont_enc_2)
                aug1 = (aug1 / aug1.norm(dim=-1, keepdim=True)).flatten(1, 2)
                aug2 = (aug2 / aug2.norm(dim=-1, keepdim=True)).flatten(1, 2)
                aug1 = model.pt_mlp(aug1)
                aug2 = model.pt_mlp2(aug2)
                c1 = aug1 @ aug2.t()
                loss += opt.lam1 * torch.diagonal(-c1).add_(1).pow_(2).sum()

            # Denoising loss
            if 'denoising' in opt.pt_tasks:
                cat_outs, con_outs = model(x_categ_enc_2, x_cont_enc_2)
                if con_outs:
                    con_outs = torch.cat(con_outs, dim=1)
                    if opt.have_xOrg:
                        l2 = criterion2(con_outs, x_cont)
                    else:
                        l2 = F.mse_loss(con_outs * (1 - train_mask_batch), x_cont * (1 - train_mask_batch), reduction='none')
                        N = (1 - train_mask_batch).sum()
                        l2 = l2.sum() / N
                else:
                    l2 = 0
                l1 = sum(criterion1(cat_outs[j], x_categ_imp[:, j]) for j in range(1, x_categ_imp.shape[-1]))
                loss += opt.lam2 * l1 + opt.lam3 * l2

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1

        # Record metrics for the epoch
        metrics_dict[missing_rate_key][fold_key]['epochs'][f'epoch_{epoch}'] = {'running_loss': running_loss}
        print(f'Epoch {epoch + 1}, Loss: {running_loss / num_batches}')

    # Save training metrics
    with open(filename, 'wb') as f:
        pickle.dump(metrics_dict, f)

    # Evaluate the model if original data is not available
    if not opt.have_xOrg:
        model.eval()
        with torch.no_grad():
            for batch in trainloader:
                x_categ, x_cont, x_categ_imp, x_cont_imp, _, cat_mask, con_mask, t_mask = \
                    [d.to(device) for d in batch]
                _, x_categ_enc_2, x_cont_enc_2 = embed_data_mask(x_categ_imp, x_cont_imp, cat_mask, con_mask, model, vision_dset)
                cat_outs, con_outs = model(x_categ_enc_2, x_cont_enc_2)
                con_outs = [x.cpu().numpy() for x in con_outs]
                cat_outs_np = [x.cpu().numpy() for x in cat_outs]
                train_mask_t.append(t_mask.cpu().numpy())
                all_predictions.append(np.concatenate(con_outs, axis=1))
                all_original_data.append(x_cont.cpu().numpy())
                all_predictions_cat.append(np.concatenate(cat_outs_np, axis=1))
                all_original_cat.append(x_categ.cpu().numpy())

        # Compute NRMSE for continuous features
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_original_data = np.concatenate(all_original_data, axis=0)
        train_mask_t = np.concatenate(train_mask_t, axis=0)
        observed_entries = 1 - train_mask_t
        mean, std = continuous_mean_std
        # Denormalize predictions and original data
        all_predictions = (all_predictions * std) + mean
        all_original_data = (all_original_data * std) + mean
        # Combine observed and missing parts
        all_predictions = (all_original_data * observed_entries) + (all_predictions * train_mask_t)
        
        mse_con = np.mean((all_original_data - all_predictions) ** 2, axis=0)
        feature_variances = np.var(all_original_data, axis=0)
        feature_variances[feature_variances == 0] = 1  # avoid divide-by-zero

        nrmse_con = np.mean(np.sqrt(mse_con) / feature_variances)
        nrmse_cat = 0.0  # Not computed in this refactor
        print('NRMSE for Continuous Features on the Train set:', nrmse_con)
        print('NRMSE for Categorical Features on the Train set:', nrmse_cat)
    else:
        nrmse_con, nrmse_cat = 0.0, 0.0

    print('END OF PRETRAINING!')
    return model, nrmse_con, nrmse_cat

import os
import argparse
import pickle
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from models import DeepIFSAC as DeepIFSAC_default

from data_openml import my_data_prep_openml, task_dset_ids, DataSetCatCon, DataSetCatCon_imputedX
from utils import count_parameters, my_classification_scores, mean_sq_error, clf_scores, imputed_data, run_mlp, train_and_test
from augmentations import embed_data_mask, add_noise

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone as sk_clone
from sklearn.pipeline import Pipeline

# Define common classifiers for downstream tasks.
common_classifiers = {
    'Logistic Regression': {
        'model': LogisticRegression(random_state=42, class_weight="balanced", max_iter=300, solver='liblinear'),
        'params': {'C': [0.8, 0.5, 1, 5, 0.01, 0.05], 'penalty': ['l1', 'l2']},
    },
    'Random Forest': {
        'model': RandomForestClassifier(n_estimators=5, class_weight="balanced", random_state=42),
        'params': {'n_estimators': list(range(10, 120, 20))},
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(class_weight='balanced', random_state=42),
        'params': {"min_samples_split": [2, 10, 20], "max_depth": [2, 5, 10]},
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'params': {"n_estimators": [50, 80, 110], "max_depth": [2, 5, 10, 15]},
    },
    'SVM Linear': {
        'model': SVC(kernel='linear', random_state=42, class_weight="balanced", probability=True),
        'params': {'C': [0.8, 0.5, 0.1, 0.05, 0.01]},
    },
    'SVM RBF': {
        'model': SVC(kernel='rbf', random_state=42, cache_size=20000, class_weight="balanced", probability=True),
        'params': {'C': [0.8, 0.5, 0.1, 0.05, 0.01], 'gamma': [0.1, 0.02, 0.3, 0.5, 0.05, 0.01]},
    },
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset_id', default=11, type=int)
    parser.add_argument('--vision_dset', action='store_true')
    parser.add_argument('--task', default='multiclass', type=str, choices=['binary', 'multiclass', 'regression'])
    parser.add_argument('--cont_embeddings', default='MLP', type=str, choices=['MLP', 'Noemb', 'pos_singleMLP'])
    parser.add_argument('--embedding_size', default=32, type=int)
    parser.add_argument('--transformer_depth', default=6, type=int)
    parser.add_argument('--attention_heads', default=8, type=int)
    parser.add_argument('--attention_dropout', default=0.1, type=float)
    parser.add_argument('--ff_dropout', default=0.1, type=float)
    parser.add_argument('--attentiontype', default='colrow', type=str,
                        choices=['col', 'colrow', 'row', 'justmlp', 'attn', 'attnmlp', 'parallel', 'rowcol', 'colrowatt'])
    parser.add_argument('--optimizer', default='AdamW', type=str, choices=['AdamW', 'Adam', 'SGD'])
    parser.add_argument('--scheduler', default='cosine', type=str, choices=['cosine', 'linear'])
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--batchsize', default=256, type=int)
    parser.add_argument('--savemodelroot', default='./bestmodels', type=str)
    parser.add_argument('--run_name', default='testrun', type=str)
    parser.add_argument('--set_seed', default=1, type=int)
    parser.add_argument('--dset_seed', default=0, type=int)
    parser.add_argument('--active_log', action='store_true')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--pretrain_epochs', default=10, type=int)
    parser.add_argument('--pretrainer', default='DeepIFSAC', type=str, choices=['DeepIFSAC', 'recon'])
    parser.add_argument('--pt_tasks', default=['denoising'], type=str, nargs='*',
                        choices=['contrastive', 'contrastive_sim', 'denoising'])
    parser.add_argument('--pt_aug', default=[], type=str, nargs='*', choices=['mixup', 'cutmix'])
    parser.add_argument('--pt_aug_lam', default=0.3, type=float)
    parser.add_argument('--mixup_lam', default=0.2, type=float)
    parser.add_argument('--missing_rate', default=0, type=float)
    parser.add_argument('--missing_type', default='mcar', type=str, choices=['mcar', 'mnar', 'mar'])
    parser.add_argument('--corruption_type', default='cutmix', type=str,
                        choices=['cutmix', 'zeroes', 'median', 'no_corruption'])
    parser.add_argument('--train_mask_prob', default=0, type=float)
    parser.add_argument('--mask_prob', default=0, type=float)
    parser.add_argument('--ssl_avail_y', default=0, type=int)
    parser.add_argument('--pt_projhead_style', default='diff', type=str, choices=['diff', 'same', 'nohead'])
    parser.add_argument('--nce_temp', default=0.7, type=float)
    parser.add_argument('--lam0', default=0.5, type=float)
    parser.add_argument('--lam1', default=10, type=float)
    parser.add_argument('--lam2', default=1, type=float)
    parser.add_argument('--lam3', default=10, type=float)
    parser.add_argument('--final_mlp_style', default='sep', type=str, choices=['common', 'sep'])
    parser.add_argument('--cuda_device', default=0, type=int)
    parser.add_argument('--use_default_model', action='store_true')
    parser.add_argument('--have_xOrg', action='store_true')
    opt = parser.parse_args()

    corruptor_settings = {
        'method': 'mcar_missing',
        'corruption_rate': 0.6,
        'missing': opt.missing_rate,
        'missing_type': opt.missing_type,
        'mice': 'LinearRegression'
    }

    print(opt)
    # Use DeepIFSAC.
    model_class = DeepIFSAC_default

    modelsave_path = os.path.join(os.getcwd(), opt.savemodelroot, opt.task, str(opt.dset_id), opt.run_name)
    if opt.task == 'regression':
        opt.dtask = 'reg'
    else:
        opt.dtask = 'clf'

    if opt.attentiontype == 'colrowatt':
        opt.pt_tasks = ['denoising']
    if opt.attentiontype == 'colrow':
        opt.pt_tasks = ['denoising', 'contrastive']

    device = torch.device(f"cuda:{opt.cuda_device}" if (opt.cuda_device >= 0 and torch.cuda.is_available()) else "cpu")
    print(f"Device is {device}.")

    torch.manual_seed(opt.set_seed)
    np.random.seed(opt.set_seed)
    os.makedirs(modelsave_path, exist_ok=True)

    if opt.active_log:
        import wandb
        if opt.pretrain:
            wandb.init(project="deepifsac_v2_all", group=opt.run_name,
                       name=f'pretrain_{opt.task}_{opt.attentiontype}_{opt.dset_id}_{opt.set_seed}')
        else:
            proj = "deepifsac_v2_all_kamal" if opt.task == 'multiclass' else "deepifsac_v2_all"
            wandb.init(project=proj, group=opt.run_name,
                       name=f'{opt.task}_{opt.attentiontype}_{opt.dset_id}_{opt.set_seed}')

    print('Downloading and processing the dataset, it might take some time.')
    cat_dims, cat_idxs, con_idxs, X_train, y_train, X_test, y_test, train_mean, train_std = my_data_prep_openml(
        opt.dset_id, opt.dset_seed, opt.task)
    continuous_mean_std = np.array([train_mean, train_std]).astype(np.float32)

    # Generate missingness on whole dataset.
    X_train_imp, train_mask = imputed_data(X_train['data'], corruptor_settings)
    X_test_imp, test_mask = imputed_data(X_test['data'], corruptor_settings)

    imp_train = np.array(X_train_imp[:, con_idxs].cpu(), dtype=np.float32)
    imp_train_mean, imp_train_std = imp_train.mean(0), imp_train.std(0)
    imp_train_std = np.where(imp_train_std < 1e-6, 1e-6, imp_train_std)
    imp_continuous_mean_std = np.array([imp_train_mean, imp_train_std]).astype(np.float32)

    # Adjust hyperparameters based on dataset.
    _, nfeat = X_train['data'].shape
    if nfeat > 100:
        opt.embedding_size = min(8, opt.embedding_size)
        opt.batchsize = min(64, opt.batchsize)
    print(nfeat, opt)
    if opt.active_log:
        wandb.config.update(opt)

    # Create datasets and dataloaders.
    train_ds = DataSetCatCon_imputedX(X_train, X_train_imp.cpu().numpy(), y_train, train_mask, cat_idxs,
                                      opt.dtask, continuous_mean_std, imp_continuous_mean_std)
    trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True, num_workers=0)
    test_ds = DataSetCatCon_imputedX(X_test, X_test_imp.cpu().numpy(), y_test, test_mask, cat_idxs,
                                     opt.dtask, continuous_mean_std, imp_continuous_mean_std)
    testloader = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False, num_workers=0)

    if opt.task == 'regression':
        y_dim = 1
    else:
        y_dim = len(np.unique(y_train['data'][:, 0]))

    # Append 1 for CLS token.
    cat_dims = np.append(np.array([1]), np.array(cat_dims)).astype(int)

    model = model_class(
        categories=tuple(cat_dims),
        num_continuous=len(con_idxs),
        dim=opt.embedding_size,
        dim_out=1,
        depth=opt.transformer_depth,
        heads=opt.attention_heads,
        attn_dropout=opt.attention_dropout,
        ff_dropout=opt.ff_dropout,
        mlp_hidden_mults=(4, 2),
        cont_embeddings=opt.cont_embeddings,
        attentiontype=opt.attentiontype,
        final_mlp_style=opt.final_mlp_style,
        y_dim=y_dim
    )
    vision_dset = opt.vision_dset

    if opt.task in ['binary', 'multiclass']:
        criterion = nn.CrossEntropyLoss().to(device)
    elif opt.task == 'regression':
        criterion = nn.MSELoss().to(device)
    else:
        raise ValueError('Task not written yet')

    model.to(device)

    if opt.pretrain:
        print('DeepIFSAC PRETRAIN')
        from pretraining import DeepIFSAC_pretrain  # use pretraining function for DeepIFSAC
        model, train_nrmse_con, train_nrmse_cat = DeepIFSAC_pretrain(
            model, cat_idxs, X_train, y_train, X_train_imp, train_mask,
            continuous_mean_std, imp_continuous_mean_std, opt, device)

    # Save pretrained model.
    path = f'./results/model_weights/{opt.dset_id}_{opt.attentiontype}_{opt.missing_type}_{opt.missing_rate}_{opt.dset_seed}_{opt.corruption_type}_model.pth'
    os.makedirs('./results/model_weights', exist_ok=True)
    torch.save(model.state_dict(), path)
    print("Pretraining Done")

    # Choose optimizer.
    if opt.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
        from utils import get_scheduler
        scheduler = get_scheduler(opt, optimizer)
    elif opt.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    elif opt.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=opt.lr)

    best_valid_auroc = best_valid_accuracy = best_test_auroc = best_test_accuracy = 0
    best_valid_rmse = 1e6
    best_valid_loss = np.inf
    best_epoch = -1

    # -----------------------------
    # Evaluate model on training set.
    # -----------------------------
    all_predictions_train = []
    all_original_data = []
    all_predictions_cat = []
    all_original_cat = []
    y_train_list = []
    train_mask_t = []
    model.eval()
    with torch.no_grad():
        for data in trainloader:
            x_categ, x_cont, x_categ_imp, x_cont_imp, y_t, cat_mask, con_mask, t_mask = [d.to(device) for d in data]
            if opt.have_xOrg:
                _, x_categ_enc_2, x_cont_enc_2 = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset)
            else:
                _, x_categ_enc_2, x_cont_enc_2 = embed_data_mask(x_categ_imp, x_cont_imp, cat_mask, con_mask, model, vision_dset)
            cat_outs, con_outs = model(x_categ_enc_2, x_cont_enc_2)
            con_outs = [x.cpu().numpy() for x in con_outs]
            cat_outs_device = [x.cpu().numpy() for x in cat_outs]
            train_mask_t.append(t_mask.cpu().numpy())
            y_train_list.append(y_t.cpu().numpy())
            all_predictions_train.append(np.concatenate(con_outs, axis=1))
            all_original_data.append(x_cont.cpu().numpy())
            all_predictions_cat.append(np.concatenate(cat_outs_device, axis=1))
            all_original_cat.append(x_categ.cpu().numpy())

    all_predictions_train = np.concatenate(all_predictions_train, axis=0)
    all_original_data = np.concatenate(all_original_data, axis=0)
    all_predictions_cat = np.concatenate(all_predictions_cat, axis=0)
    all_original_cat = np.concatenate(all_original_cat, axis=0)
    train_mask_t = np.concatenate(train_mask_t, axis=0)
    y_train_arr = np.concatenate(y_train_list, axis=0)
    observed_entries_train = 1 - train_mask_t

    mean, std = continuous_mean_std
    mean_imp, std_imp = imp_continuous_mean_std
    all_predictions_train = torch.tensor((all_predictions_train * std_imp) + mean_imp)
    all_original_data = torch.tensor((all_original_data * std) + mean)
    all_predictions_train = (all_original_data * observed_entries_train) + (all_predictions_train * train_mask_t)

    # -----------------------------
    # Evaluate model on test set.
    # -----------------------------
    all_predictions_test = []
    all_original_data_test = []
    all_predictions_cat_test = []
    all_original_cat_test = []
    y_test_list = []
    test_mask_t = []
    with torch.no_grad():
        for data in testloader:
            x_categ, x_cont, x_categ_imp, x_cont_imp, y_t, cat_mask, con_mask, t_mask = [d.to(device) for d in data]
            _, x_categ_enc_2, x_cont_enc_2 = embed_data_mask(x_categ_imp, x_cont_imp, cat_mask, con_mask, model, vision_dset)
            cat_outs, con_outs = model(x_categ_enc_2, x_cont_enc_2)
            con_outs = [x.cpu().numpy() for x in con_outs]
            cat_outs_device = [x.cpu().numpy() for x in cat_outs]
            y_test_list.append(y_t.cpu().numpy())
            test_mask_t.append(t_mask.cpu().numpy())
            all_predictions_test.append(np.concatenate(con_outs, axis=1))
            all_original_data_test.append(x_cont.cpu().numpy())
            all_predictions_cat_test.append(np.concatenate(cat_outs_device, axis=1))
            all_original_cat_test.append(x_categ.cpu().numpy())

    all_predictions_test = np.concatenate(all_predictions_test, axis=0)
    all_original_data_test = np.concatenate(all_original_data_test, axis=0)
    all_predictions_cat_test = np.concatenate(all_predictions_cat_test, axis=0)
    all_original_cat_test = np.concatenate(all_original_cat_test, axis=0)
    y_test_arr = np.concatenate(y_test_list, axis=0)
    test_mask_t = np.concatenate(test_mask_t, axis=0)
    observed_entries = torch.tensor(1 - test_mask_t)

    all_predictions_test = torch.tensor((all_predictions_test * std) + mean)
    all_original_data_test = torch.tensor((all_original_data_test * std) + mean)
    all_predictions_test = (all_original_data_test * observed_entries) + (all_predictions_test * test_mask_t)

    mse_con = torch.mean((all_original_data_test - all_predictions_test) ** 2, dim=0)
    feature_means = torch.where(torch.mean(all_original_data_test, dim=0) == 0,
                                torch.ones_like(torch.mean(all_original_data_test, dim=0)),
                                torch.mean(all_original_data_test, dim=0))
    feature_variances = torch.where(torch.var(all_original_data_test, dim=0) == 0,
                                    torch.ones_like(torch.var(all_original_data_test, dim=0)),
                                    torch.var(all_original_data_test, dim=0))
    nrmse_con = torch.mean(torch.sqrt(mse_con) / feature_variances)
    nrmse_cat = torch.tensor(0)

    print('NRMSE for Continuous Features on the Test set:', nrmse_con.item())
    print('NRMSE for Categorical Features on the Test set:', nrmse_cat.item())

    # ------------------------------------------------------------------
    # Downstream Classification: Separate LR/GBT results and MLP training.
    # ------------------------------------------------------------------
    # Create imputed feature sets by concatenating continuous and categorical features.
    imp_train_mean, imp_train_std = np.array(all_predictions_train[:, con_idxs], dtype=np.float32).mean(0), \
                                    np.array(all_predictions_train[:, con_idxs], dtype=np.float32).std(0)
    all_predictions_train_norm = (all_predictions_train - imp_train_mean) / imp_train_std
    all_predictions_downstream_test = (all_predictions_test - imp_train_mean) / imp_train_std

    imputed_train = torch.cat([torch.from_numpy(all_predictions_train_norm.detach().cpu().numpy()),
                               torch.from_numpy(all_predictions_cat[:, 1:])], dim=1).to(device)
    imputed_test = torch.cat([all_predictions_downstream_test,
                              torch.from_numpy(all_predictions_cat_test[:, 1:])], dim=1).to(device)
    x_dim = imputed_train.shape[1]
    output_size = np.unique(y_train_arr).shape[0]

    # First, train common classifiers (e.g., LR and GBT).
    clfs = [common_classifiers['Logistic Regression'],common_classifiers['Gradient Boosting']]

    lr, gbt = train_and_test(clfs,
                             (imputed_train.cpu().detach().numpy(), np.squeeze(y_train_arr)),
                             (imputed_test.cpu().detach().numpy(), np.squeeze(y_test_arr)))
    lr, gbt = lr * 100, gbt * 100

    # Next, train a separate MLP.
    from models.model import simple_MLP
    mlpfory = simple_MLP([x_dim, 1000, output_size]).to(device)
    criterion_n = nn.CrossEntropyLoss().to(device)
    cls_model, output_mlp = run_mlp(mlpfory, imputed_train.to(device),
                                    torch.from_numpy(y_train_arr).squeeze().to(device),
                                    imputed_test.to(device),
                                    torch.from_numpy(y_test_arr).squeeze().to(device),
                                    criterion_n, opt.batchsize, opt.epochs, opt)

    print('END OF FINETUNING!')
    print(train_nrmse_con, train_nrmse_cat, nrmse_con.item(), nrmse_cat.item())
    print(f'best acc LR: {lr:.2f} acc GBT: {gbt:.2f}')
    print(f'best valid f1: {0.0} test f1: {output_mlp:.2f}')


if __name__ == '__main__':
    main()
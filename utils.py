import torch
from sklearn.metrics import roc_auc_score, mean_squared_error, f1_score
import numpy as np
from augmentations import embed_data_mask,add_noise
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score

def make_default_mask(x):
    mask = np.ones_like(x)
    mask[:,-1] = 0
    return mask

def tag_gen(tag,y):
    return np.repeat(tag,len(y['data']))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  

def get_scheduler(args, optimizer):
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                      milestones=[args.epochs // 2.667, args.epochs // 1.6, args.epochs // 1.142], gamma=0.1)
    return scheduler

def imputations_acc_justy(model,dloader,device):
    model.eval()
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model)
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:,model.num_categories-1,:]
            y_outs = model.mlpfory(y_reps)
            # import ipdb; ipdb.set_trace()   
            y_test = torch.cat([y_test,x_categ[:,-1].float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(m(y_outs), dim=1).float()],dim=0)
            prob = torch.cat([prob,m(y_outs)[:,-1].float()],dim=0)
     
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())
    return acc, auc


def multiclass_acc_justy(model,dloader,device):
    model.eval()
    vision_dset = True
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:,model.num_categories-1,:]
            y_outs = model.mlpfory(y_reps)
            # import ipdb; ipdb.set_trace()   
            y_test = torch.cat([y_test,x_categ[:,-1].float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(m(y_outs), dim=1).float()],dim=0)
     
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    return acc, 0


def classification_scores(model, dloader, device, task,vision_dset):
    model.eval()
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)           
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:,0,:]
            y_outs = model.mlpfory(y_reps)
            # import ipdb; ipdb.set_trace()   
            y_test = torch.cat([y_test,y_gts.squeeze().float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(y_outs, dim=1).float()],dim=0)
            if task == 'binary':
                prob = torch.cat([prob,m(y_outs)[:,-1].float()],dim=0)
     
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    auc = 0
    if task == 'binary':
        auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())

    # f1 = f1_score(y_pred.cpu().numpy(), y_test.cpu().numpy(), average='weighted')
    acc_cpu = acc.cpu().numpy()
    return acc_cpu, auc

def my_classification_scores(model, dloader, device, task,vision_dset,opt):
    
    model.eval()
    pt_aug_dict = {
        'noise_type' : opt.pt_aug,
        'lambda' : opt.pt_aug_lam
    }
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            # x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)
            x_categ, x_cont, x_categ_imp, x_cont_imp,y_t ,cat_mask, con_mask, t_mask = [data[i].to(device) for i in range(8)]
            
            # x_categ_corr, x_cont_corr = add_noise(x_categ,x_cont, noise_params = pt_aug_dict, mr = opt.missing_rate, mt= opt.missing_type)  #add corruption corresponding to the missingness
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ_imp, x_cont_imp, cat_mask, con_mask,model,vision_dset)           
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:,0,:]
            y_outs = model.mlpfory(y_reps)
            # import ipdb; ipdb.set_trace()   
            y_test = torch.cat([y_test,y_gts.squeeze().float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(y_outs, dim=1).float()],dim=0)
            if task == 'binary':
                prob = torch.cat([prob,m(y_outs)[:,-1].float()],dim=0)
     
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    auc = 0
    # if task == 'binary':
    #     auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())

    f1 = f1_score(y_pred.cpu().numpy(), y_test.cpu().numpy(), average='weighted')
    acc_cpu = acc.cpu().numpy()
    # return f1, y_pred, y_test
    return f1*100, acc_cpu

def clf_scores(model, test_dataloader, opt):
    import os
    import pickle

    model.eval()
    predictions = []
    labels = []
    probabilities = []


    with torch.no_grad():
        for inputs, targets in test_dataloader:
            outputs = model(inputs)
            # print(outputs)
            y_pred = torch.argmax(outputs, 1)
            y_prob = torch.softmax(outputs, dim=1)
            # print('proba: ', y_prob)
            predictions.extend(y_pred.cpu().numpy())
            labels.extend(targets.cpu().numpy())
            probabilities.extend(y_prob.cpu().numpy())
    
    
    directory = f'./results/preds'
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    filename = f'{directory}/{opt.dset_id}_{opt.attentiontype}_{opt.missing_type}_{opt.corruption_type}_model_predictions_labels.pkl'

    # Initialize or load the dictionary
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            data_dict = pickle.load(f)
    else:
        data_dict = {}

#     # Update the dictionary with the new fold data
#     data_dict[f'fold_{opt.dset_seed}'] = {'predictions': predictions, 'labels': labels}

#     # Save the updated dictionary back to the file
#     with open(filename, 'wb') as f:
#         pickle.dump(data_dict, f)
        
    fold_key = f'fold_{opt.dset_seed}'
    missing_key = str(opt.missing_rate)

    if fold_key not in data_dict:
        data_dict[fold_key] = {}

    data_dict[fold_key][missing_key] = {'predictions': predictions, 'labels': labels, 'probabilities': probabilities}

    with open(filename, 'wb') as f:
        pickle.dump(data_dict, f)
    # print(predictions)
    # print(labels)
    labels = np.array(labels)
    probabilities = np.array(probabilities)
    # print(probabilities.shape, labels.shape)
    f1 = f1_score(predictions, labels, average='weighted')
    if probabilities.shape[-1] == 2:
        probabilities = probabilities[:,1]
    auc = roc_auc_score(labels, probabilities, multi_class='ovr')
    if opt.attentiontype == 'midaspy':
        return f1, auc
    else:
        return f1



def mean_sq_error(model, dloader, device, vision_dset):
    model.eval()
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)           
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:,0,:]
            y_outs = model.mlpfory(y_reps)
            y_test = torch.cat([y_test,y_gts.squeeze().float()],dim=0)
            y_pred = torch.cat([y_pred,y_outs],dim=0)
        # import ipdb; ipdb.set_trace() 
        rmse = mean_squared_error(y_test.cpu(), y_pred.cpu(), squared=False)
        return rmse
    
    
    
def recreate_empty_file(filename):
    """
    Checks if the specified file is empty, and if so, deletes it and creates a new empty file with the same name.

    Parameters:
    - filename (str): The path to the file to check and recreate if empty.
    """
    
    # Check if the file exists to avoid FileNotFoundError
    if os.path.exists(filename):
        # Check if the file is empty by looking at its size
        if os.path.getsize(filename) == 0:
            print(f"File {filename} is empty. Deleting and creating a new one.")
            os.remove(filename)  # Delete the empty file
            
            # Create a new, empty file with the same name
            with open(filename, 'w') as f:
                pass  # 'pass' simply creates an empty block, resulting in an empty file
            print(f"New empty file {filename} created.")
        else:
            print(f"File {filename} is not empty.")
    else:
        # If the file doesn't exist, simply create a new one
        with open(filename, 'w') as f:
            pass  # Create a new empty file
        print(f"File {filename} did not exist and was created.")
        
        
def imputed_data_main(data, settings):
    from corruptor import Corruptor
    corruptor_x = Corruptor(data, settings)
    data, mask = corruptor_x(torch.tensor(data))
    # median = torch.median(data[~torch.isnan(data)])
    median = torch.nanmedian(data, dim=0).values
    X_train_imp = torch.where(torch.isnan(data), median, data)
    return X_train_imp, mask

def imputed_data(data, settings, opt = None):
    # print('initiaal; data', data)
    from corruptor import Corruptor
    if opt is not None:
        corruptor_settings ={
                    'method': 'draw',
                    'corruption_rate': 0.6,
                    'missing': opt.missing_rate,
                    'missing_type': 'mcar', #opt.missing_type
                    'mice': 'LinearRegression'
                }
        data = torch.tensor(data)
        corruptor= Corruptor(data, corruptor_settings)
        X_train_imp, mask = corruptor(data)
        # X_train_imp = X_train_imp.cpu().numpy()
        X_train_imp = torch.tensor(X_train_imp)
        # print(X_train_imp)
    else:
        corruptor_x = Corruptor(data, settings)
        data, mask = corruptor_x(torch.tensor(data))
        # median = torch.median(data[~torch.isnan(data)])
        median = torch.nanmedian(data, dim=0).values
        X_train_imp = torch.where(torch.isnan(data), median, data)

    return X_train_imp, mask

def train_and_test(clf, train, test):
    train_x1, train_y1 = train
    # train_x1 = train_x1.cpu()
    # valid_x, valid_y = valid
    test_x1, test_y1 = test
    # print(test_y1.shape)
    # test_x1 = test_x1.cpu().detach().numpy()
    # test_y1 = test_y1.cpu().detach().numpy()
    # train_x1 = train_x1.cpu().detach().numpy()
    # train_y1 = train_y1.cpu().detach().numpy()

    best_model = False
    # best_params = False
    # best_val = 0

    f1_scores = []
    y_preds = []
    try:
        for m in clf:
            best_model = m['model']
            # best_model.set_params(**best_params)
            best_model.fit(train_x1, train_y1)
            # print("y_pred test ",test_x.shape)
            y_pred = best_model.predict(test_x1)

            score = f1_score(test_y1, y_pred, average='weighted')
            f1_scores.append(score)
            # y_preds.append(y_pred)
        # print(f1_scores)
        return f1_scores
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return 0 

def run_mlp(model, train_data, train_y, test_data, test_y, criterion, batch_size, num_epochs,opt):
    from torch.utils.data import DataLoader, TensorDataset
    import torch.optim as optim
        
    learning_rate = 0.0001
    train_data = train_data.float()
    train_y = train_y.long()
    test_data = test_data.float()
    test_y = test_y.long()
    # Convert to TensorDataset and DataLoader for batch processing
    train_dataset = TensorDataset(train_data, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(test_data, test_y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Convert inputs and labels to the correct type
            inputs = inputs.float()
            labels = labels.long()  # Use long for CrossEntropyLoss
            # print(labels)
            # Forward pass
            outputs = model(inputs)
            # print(outputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluation
    #model.eval()  # Set the model to evaluation mode
    #correct = 0
    #total = 0
    #with torch.no_grad():  # No gradient needed for evaluation
    #    outputs = model(test_data)
    #    y_pred = torch.argmax(outputs, 1)
    #    f1 = f1_score(test_y.cpu().numpy(), y_pred.cpu().numpy(), average='weighted') 
    #    #This works just fine, but when we will have large test set might face OOM error
        
        
        
        #     inputs = inputs.float()
        #     labels = labels.long()
        #     # print(inputs.shape)
        #     outputs = model(inputs)
        #     # print(outputs.shape)
        #     _, predicted = torch.max(outputs.data, 1)  # Get the index of the max log-probability
        #     # exit()
        #     total += labels.size(0)
        #     correct += (predicted == labels).sum().item()

    # accuracy = correct / total
    
    if opt.attentiontype == 'midaspy':
        f1, auc = clf_scores(model, test_loader, opt)
        # print(f'F1 score of the model on the test set: {f1}%')
        # print(type(f1))
        return model, round(f1*100,2), auc
    else:
        f1 = clf_scores(model, test_loader, opt)
        # print(f'F1 score of the model on the test set: {f1}%')
        # print(type(f1))
        return model, round(f1*100,2)


def get_imputed_data(model, dataloader):
    all_predictions = []
    all_original_data = []
    all_predictions_cat = []
    all_original_cat = []
    y_list = []

    model.eval()
    # print(model.cat_mask_offset)
    # Iterate through the DataLoader to make predictions on the whole dataset
    pt_aug_dict = {
            'noise_type' : opt.pt_aug,
            'lambda' : opt.pt_aug_lam
        }
    # torch.manual_seed(opt.set_seed)
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            # x_categ, x_cont, _, cat_mask, con_mask = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)
            x_categ_main, x_cont_main, x_categ, x_cont, y_t, cat_mask, con_mask, t_mask = [data[i].to(device) for i in range(8)]

            # x_categ_corr, x_cont_corr = add_noise(x_categ,x_cont, noise_params = pt_aug_dict, mr = opt.missing_rate, mt= opt.missing_type)  #add corruption corresponding to the missingness
            _ , x_categ_enc_2, x_cont_enc_2 = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)

            # Assuming 'model' is your pretrained model
            cat_outs, con_outs = model(x_categ_enc_2, x_cont_enc_2)

            # Append predictions and original data to the lists
            # print(len(con_outs))
            y_list.append(y_t.cpu().numpy())
            con_outs = [x.cpu().numpy() for x in con_outs]
            cat_outs_device = [x.cpu().numpy() for x in cat_outs]

            all_predictions.append(np.concatenate(con_outs, axis=1))
            all_original_data.append(x_cont.cpu().numpy())
            all_predictions_cat.append(np.concatenate(cat_outs_device, axis=1))
            all_original_cat.append(x_categ.cpu().numpy())
            # print(np.concatenate(con_outs, axis=1).shape, x_cont.shape)

    # Concatenate predictions and original data into tensors
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_original_data = np.concatenate(all_original_data, axis=0)
    all_predictions_cat = np.concatenate(all_predictions_cat, axis=0)
    all_original_cat = np.concatenate(all_original_cat, axis=0)
    all_y_list = np.concatenate(y_list, axis=0)
    return all_predictions, all_predictions_cat, all_original_data, all_original_cat, all_y_list




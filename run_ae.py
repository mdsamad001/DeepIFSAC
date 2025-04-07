import os
import pickle
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--job_split', default=0, type=int)
parser.add_argument('--job_id', default=0, type=int)
parser.add_argument('--cuda_device', default=0, type=int) # passed to trainer
parser.add_argument('--use_default_model', action='store_true') # passed to trainer
parser.add_argument('--have_xOrg', action='store_true') # passed to trainer

parser.add_argument('--missing_type', default='mcar', type=str,
                     choices=['mcar', 'mnar','mar'])
parser.add_argument('--attentiontype', default='colrow', type=str,
                     choices=['col', 'rowcol','colrow', 'parallel', 'colrowatt', 'row']) 

opt = parser.parse_args()
import json

def save_var(var, filename, do_print=True):
    '''Saving the objects in JSON format.'''
    try:
        with open(filename, 'w') as f:  # Open in write mode
            json.dump(var, f, indent=4)  # Serialize with indentation for readability
    except Exception as e:
        if do_print:
            print('Could not save', filename, 'because', e)
        return False
    
    return True


# def load_var(filename, do_print = True):
#     '''Getting back the objects:'''
#     try:
#         with open(filename, 'rb') as f:  # Python 3: open(..., 'rb')
#             var = pickle.load(f)
#     except Exception as e:
#         do_print and print('Could not load', filename, 'because', e)
#         return False

#     return var
def load_var(filename, do_print=True):
    '''Getting back the objects from JSON format.'''
    try:
        with open(filename, 'r') as f:  # Open in read mode
            var = json.load(f)  # Deserialize
    except Exception as e:
        if do_print:
            print('Could not load', filename, 'because', e)
        return False

    return var

def get_results(dataset):
    valid_dict = {0.1:[], 0.3: [], 0.5:[], 0.7: [], 0.9:[]}
    test_dict = {0.1:[], 0.3: [], 0.5:[], 0.7: [], 0.9:[]}
    lr_dict = {0.1:[], 0.3: [], 0.5:[], 0.7: [], 0.9:[]}
    gbt_dict = {0.1:[], 0.3: [], 0.5:[], 0.7: [], 0.9:[]}
    imputation_dict = {0.1:[], 0.3: [], 0.5:[], 0.7: [], 0.9:[]}
    # 0.1,0.3,0.5
    for mr in [0.7, 0.9]:
        valid_list = []
        test_list = []
        imputation = []
        lr_list = []
        gbt_list = []
        for i in range(5):
            r  = os.popen((f'python my_train.py --dset_id {dataset} --task multiclass --attentiontype {opt.attentiontype} '
                        '--pt_aug cutmix '
                        '--pretrain --pretrain_epochs 1000  --epochs 1 --batchsize 128 '
                        f'--dset_seed {i} --cuda_device {opt.cuda_device} {"--use_default_model" if opt.use_default_model else ""} --missing_rate {mr} --missing_type {opt.missing_type} {"--have_xOrg" if opt.have_xOrg else ""} '
                        )).readlines()


            # print(r[:-1])
            # print(r)
            # exit()
            imputation_error = r[-3].split()
            imputation.append(imputation_error)
            # print(imputation_error)
            splits = r[-1].replace('\n', '').split(' ')
            # print('splits', splits)
            
            valid, test = float(splits[-4]), float(splits[-1])
            valid_list.append(valid)
            test_list.append(test)
            
            gbt_splits = r[-2].replace('\n', '').split(' ')
            # print('gbt splits', gbt_splits)
            lr, gbt = float(gbt_splits[-4]), float(gbt_splits[-1])
            lr_list.append(lr)
            gbt_list.append(gbt)
            # print(valid, test)
            
            

        valid_dict[mr]=valid_list
        test_dict[mr]=test_list
        imputation_dict[mr]=imputation
        lr_dict[mr] = lr_list
        gbt_dict[mr] = gbt_list

        
    # return valid_list, test_list, imputation
    return valid_dict, test_dict, imputation_dict, lr_dict, gbt_dict
#  11, 37, 54, , 
datasets = [11, 37, 54, 187, 1464, 1049, 1050, 1067, 1068, 1497, 40982, 458]
# datasets = [458]
datasets = datasets if opt.job_split == 0 else datasets[opt.job_id::opt.job_split]
fname = f"saint_{opt.missing_type}_{opt.attentiontype}" if opt.use_default_model else "saint-ae"
from time import time
for i,d in enumerate(datasets, 1):
    # if d in [1485, 4134]: continue
    
    print(f'doing {d}; {i}/{len(datasets)}')
    file_path = f'./saved_vars/{fname}_{d}.json'
    # file_path_new = f'./saved_vars_gbt_new/{fname}_{d}.pkl'

    # r = load_var(file_path, do_print = False)
    
    r= False
    if not r:
        starting_time = time()
        r = get_results(d)
        time_taken = time() - starting_time
        r1 = r[0:5]+tuple([time_taken])
        # r2 = r[3:5]
#         print(r1)
#         print('=======')
        
#         print(r2)
        # print(len(r))
        # print('Executing Dataset ', d)
        save_var(r1, file_path, do_print=False)
        
#         save_var(r2, file_path_new, do_print=False)
        

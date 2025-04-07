import openml
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold

# import openml
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, LabelBinarizer
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline


def do_nothing(*args, **kwargs):
    return None

openml.datasets.functions._get_dataset_parquet = do_nothing
def simple_lapsed_time(text, lapsed):
    hours, rem = divmod(lapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(text+": {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


def task_dset_ids(task):
    dataset_ids = {
        'binary': [1487,44,1590,42178,1111,31,42733,1494,1017,4134],
        'multiclass': [188, 1596, 4541, 40664, 40685, 40687, 40975, 41166, 41169, 42734],
        'regression':[541, 42726, 42727, 422, 42571, 42705, 42728, 42563, 42724, 42729]
    }

    return dataset_ids[task]

def concat_data(X,y):
    # import ipdb; ipdb.set_trace()
    return pd.concat([pd.DataFrame(X['data']), pd.DataFrame(y['data'][:,0].tolist(),columns=['target'])], axis=1)


def data_split(X,y,nan_mask,indices):
    x_d = {
        'data': X.values[indices],
        'mask': nan_mask.values[indices]
    }
    
    if x_d['data'].shape != x_d['mask'].shape:
        raise'Shape of data not same as that of nan mask!'
        
    y_d = {
        'data': y[indices].reshape(-1, 1)
    } 
    return x_d, y_d


def data_prep_openml(ds_id, seed, task, datasplit=[.65, .15, .2]):
    
    np.random.seed(seed) 
    dataset = openml.datasets.get_dataset(ds_id)
    
    X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)
    if ds_id == 42178:
        categorical_indicator = [True, False, True,True,False,True,True,True,True,True,True,True,True,True,True,True,True,False, False]
        tmp = [x if (x != ' ') else '0' for x in X['TotalCharges'].tolist()]
        X['TotalCharges'] = [float(i) for i in tmp ]
        y = y[X.TotalCharges != 0]
        X = X[X.TotalCharges != 0]
        X.reset_index(drop=True, inplace=True)
        # print(y.shape, X.shape)
    if ds_id in [42728,42705,42729,42571]:
        # import ipdb; ipdb.set_trace()
        X, y = X[:50000], y[:50000]
        X.reset_index(drop=True, inplace=True)
    categorical_columns = X.columns[list(np.where(np.array(categorical_indicator)==True)[0])].tolist()
    cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))

    cat_idxs = list(np.where(np.array(categorical_indicator)==True)[0])
    con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))

    for col in categorical_columns:
        X[col] = X[col].astype("object")

    X["Set"] = np.random.choice(["train", "valid", "test"], p = datasplit, size=(X.shape[0],))

    train_indices = X[X.Set=="train"].index
    valid_indices = X[X.Set=="valid"].index
    test_indices = X[X.Set=="test"].index

    X = X.drop(columns=['Set'])
    temp = X.fillna("MissingValue")
    nan_mask = temp.ne("MissingValue").astype(int)
    
    cat_dims = []
    for col in categorical_columns:
    #     X[col] = X[col].cat.add_categories("MissingValue")
        X[col] = X[col].fillna("MissingValue")
        l_enc = LabelEncoder() 
        X[col] = l_enc.fit_transform(X[col].values)
        cat_dims.append(len(l_enc.classes_))
    for col in cont_columns:
    #     X[col].fillna("MissingValue",inplace=True)
        X.fillna(X.loc[train_indices, col].mean(), inplace=True)
    y = y.values
    if task != 'regression':
        l_enc = LabelEncoder() 
        y = l_enc.fit_transform(y)
    X_train, y_train = data_split(X,y,nan_mask,train_indices)
    X_valid, y_valid = data_split(X,y,nan_mask,valid_indices)
    X_test, y_test = data_split(X,y,nan_mask,test_indices)

    train_mean, train_std = np.array(X_train['data'][:,con_idxs],dtype=np.float32).mean(0), np.array(X_train['data'][:,con_idxs],dtype=np.float32).std(0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)
    # import ipdb; ipdb.set_trace()
    return cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std


def my_data_prep_openml(ds_id, seed, task):
    print(f'dataseed = {seed}')
    
    np.random.seed(seed) 
    if ds_id >= 0:
        dataset = openml.datasets.get_dataset(ds_id)

        X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)
        # print('target', y)
        if ds_id == 42178:
            categorical_indicator = [True, False, True,True,False,True,True,True,True,True,True,True,True,True,True,True,True,False, False]
            tmp = [x if (x != ' ') else '0' for x in X['TotalCharges'].tolist()]
            X['TotalCharges'] = [float(i) for i in tmp ]
            y = y[X.TotalCharges != 0]
            X = X[X.TotalCharges != 0]
            X.reset_index(drop=True, inplace=True)
            # print(y.shape, X.shape)
        if ds_id in [42728,42705,42729,42571]:
            # import ipdb; ipdb.set_trace()
            X, y = X[:50000], y[:50000]
            X.reset_index(drop=True, inplace=True)
        categorical_columns = X.columns[list(np.where(np.array(categorical_indicator)==True)[0])].tolist()
        cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))
    
    elif ds_id == -1:
        df = pd.read_csv("./dataset/C001_FakeHypotension.csv", index_col=0)
        df = df.query('Timepoints==19').reset_index()
        X = df.drop(columns=['PatientID','Timepoints','vasopressors'])
        y = df.vasopressors
        categorical_indicator = [False for i in df.columns]
        categorical_columns = df.columns[categorical_indicator]
    elif ds_id == -2:
        
        df = pd.read_csv("./dataset/C001_FakeSepsis.csv", index_col=0)
        df = df.query('Timepoints==19').reset_index()
        X = df.drop(columns=['PatientID','Timepoints','ReAd']) 
        y = df.ReAd
        categorical_indicator = [False for i in df.columns]
        categorical_columns = df.columns[categorical_indicator]
    
    cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))
    cat_idxs = list(np.where(np.array(categorical_indicator)==True)[0])
    con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))
    # print(cat_idxs, con_idxs, 'cat, con')
    for col in categorical_columns:
        X[col] = X[col].astype("object")

    # X["Set"] = np.random.choice(["train", "valid", "test"], p = datasplit, size=(X.shape[0],))
    y = y.values
    if task != 'regression':
        l_enc = LabelEncoder() 
        y = l_enc.fit_transform(y)
    row_idx = np.arange(X.shape[0])
    cv = StratifiedKFold(n_splits = 5, random_state=42, shuffle = True)
    k_folds_list = list(cv.split(row_idx, y))
    # train_valid_indices, test_indices = train_test_split(row_idx, test_size=1/5, stratify=y, random_state=seed)
    train_indices, test_indices = k_folds_list[seed]
    # train_indices, valid_indices = train_test_split(train_valid_indices, 
    #                                                 test_size=1/8, 
    #                                                 stratify=y[train_valid_indices], 
    #                                                 random_state=42)

    # train_indices = X[X.Set=="train"].index
    # valid_indices = X[X.Set=="valid"].index
    # test_indices = X[X.Set=="test"].index

    # X = X.drop(columns=['Set'])
    temp = X.fillna("MissingValue")
    nan_mask = temp.ne("MissingValue").astype(int)
    
    cat_dims = []
    for col in categorical_columns:
    #     X[col] = X[col].cat.add_categories("MissingValue")
        X[col] = X[col].fillna("MissingValue")
        l_enc = LabelEncoder() 
        X[col] = l_enc.fit_transform(X[col].values)
        cat_dims.append(len(l_enc.classes_))
    for col in cont_columns:
    #     X[col].fillna("MissingValue",inplace=True)
        X.fillna(X.loc[train_indices, col].mean(), inplace=True)
    
    X_train, y_train = data_split(X,y,nan_mask,train_indices)
    # X_valid, y_valid = data_split(X,y,nan_mask,valid_indices)
    X_test, y_test = data_split(X,y,nan_mask,test_indices)

    train_mean, train_std = np.array(X_train['data'][:,con_idxs],dtype=np.float32).mean(0), np.array(X_train['data'][:,con_idxs],dtype=np.float32).std(0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)
    # import ipdb; ipdb.set_trace()
    test_mean = np.array(X_test['data'][:, con_idxs], dtype=np.float32).mean(0)
    # print(test_mean)
    
    # return cat_dims, cat_idxs, con_idxs, X_train, y_train, X_test, y_test, train_mean, train_std
    return cat_dims, cat_idxs, con_idxs, X_train, y_train, X_test, y_test, train_mean, train_std
class DataSetCatCon(Dataset):
    def __init__(self, X, Y, cat_cols,task='clf',continuous_mean_std=None):
        
        cat_cols = list(cat_cols)
        X_mask =  X['mask'].copy()
        X = X['data'].copy()
        con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        self.X1 = X[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2 = X[:,con_cols].copy().astype(np.float32) #numerical columns
        self.X1_mask = X_mask[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2_mask = X_mask[:,con_cols].copy().astype(np.int64) #numerical columns
        if task == 'clf':
            self.y = Y['data']#.astype(np.float32)
        else:
            self.y = Y['data'].astype(np.float32)
        self.cls = np.zeros_like(self.y,dtype=int)
        self.cls_mask = np.ones_like(self.y,dtype=int)
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # X1 has categorical data, X2 has continuous
        return np.concatenate((self.cls[idx], self.X1[idx])), self.X2[idx],self.y[idx], np.concatenate((self.cls_mask[idx], self.X1_mask[idx])), self.X2_mask[idx]

class DataSetCatCon_imputedX(Dataset):
    def __init__(self, X, imputed_x, Y,t_mask, cat_cols,task='clf',continuous_mean_std=None, imp_continuous_mean_std = None):
        
        cat_cols = list(cat_cols)
        X_mask =  X['mask'].copy()
        X = X['data'].copy()
        con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        self.X1 = X[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2 = X[:,con_cols].copy().astype(np.float32) #numerical columns
        self.X1_mask = X_mask[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2_mask = X_mask[:,con_cols].copy().astype(np.int64) #numerical columns
        
        # imp_X_mask =  X['mask'].copy()
        imp_X = imputed_x.copy()
        self.t_mask = t_mask
        # con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        self.imp_X1 = imp_X[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.imp_X2 = imp_X[:,con_cols].copy().astype(np.float32) #numerical columns
        # self.X1_mask = X_mask[:,cat_cols].copy().astype(np.int64) #categorical columns
        # self.X2_mask = X_mask[:,con_cols].copy().astype(np.int64) #numerical columns
        
        if task == 'clf':
            self.y = Y['data']#.astype(np.float32)
        else:
            self.y = Y['data'].astype(np.float32)
        self.cls = np.zeros_like(self.y,dtype=int)
        self.cls_mask = np.ones_like(self.y,dtype=int)
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std
        if imp_continuous_mean_std is not None:
            imp_mean, imp_std = imp_continuous_mean_std
            self.imp_X2 = (self.imp_X2 - imp_mean) / imp_std

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # X1 has categorical data, X2 has continuous
        return (
            np.concatenate((self.cls[idx], self.X1[idx])), self.X2[idx],
            np.concatenate((self.cls[idx], self.imp_X1[idx])), self.imp_X2[idx],
            self.y[idx], np.concatenate((self.cls_mask[idx], self.X1_mask[idx])), self.X2_mask[idx], self.t_mask[idx])
    
    
class DataSetCatCon_imputed_testX(Dataset):
    def __init__(self, X, imputed_x, Y, t_mask, cat_cols,task='clf',continuous_mean_std=None, imp_continuous_mean_std = None):
        
        cat_cols = list(cat_cols)
        X_mask =  X['mask'].copy()
        X = X['data'].copy()
        con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        self.X1 = X[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2 = X[:,con_cols].copy().astype(np.float32) #numerical columns
        self.X1_mask = X_mask[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2_mask = X_mask[:,con_cols].copy().astype(np.int64) #numerical columns
        
        # imp_X_mask =  X['mask'].copy()
        imp_X = imputed_x.copy()
        # con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        self.imp_X1 = imp_X[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.imp_X2 = imp_X[:,con_cols].copy().astype(np.float32) #numerical columns
        # self.X1_mask = X_mask[:,cat_cols].copy().astype(np.int64) #categorical columns
        # self.X2_mask = X_mask[:,con_cols].copy().astype(np.int64) #numerical columns
        
        if task == 'clf':
            self.y = Y['data']#.astype(np.float32)
        else:
            self.y = Y['data'].astype(np.float32)
        self.t_mask = t_mask
        self.cls = np.zeros_like(self.y,dtype=int)
        self.cls_mask = np.ones_like(self.y,dtype=int)
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std
        if imp_continuous_mean_std is not None:
            imp_mean, imp_std = imp_continuous_mean_std
            self.imp_X2 = (self.imp_X2 - imp_mean) / imp_std

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # X1 has categorical data, X2 has continuous
        # return (
        #     np.concatenate((self.cls[idx], self.X1[idx])), self.X2[idx],
        #     np.concatenate((self.cls[idx], self.imp_X1[idx])), self.imp_X2[idx],
        #     self.y[idx], np.concatenate((self.cls_mask[idx], self.X1_mask[idx])), self.X2_mask[idx])
        return (np.concatenate((self.cls[idx], self.imp_X1[idx])), self.imp_X2[idx], self.y[idx], np.concatenate((self.cls_mask[idx], self.X1_mask[idx])), self.X2_mask[idx], self.t_mask)
import torch

import numpy as np
import pandas as pd

from itertools import product,cycle
from functools import partial

from missingness.sampler import mar_sampling, mcar_sampling, mnar_sampling
import MICE.micegradient.micegradient as mg
from sklearn.impute import KNNImputer

# need to remove unwanted imports

default_settings = {
    'method': 'pass',          # 'pass', 'noise' | 'draw' | 'sample' | 'knn' | 'mice'
    'corruption_rate': .6,      # 0.6 or between 0-1; fraction of features to corrupt (not used for mice/knn)
    'missing': .2,              # 0.2 between 0-1 float;  % of missingness
    'missing_type': 'mcar',     # 'mcar' | 'mnar' | 'mar'
    'mice': 'LinearRegression', # 'LinearRegression' | 'DecisionTree' | others...
}
# mask_arr = None

class Corruptor:

    def __init__(self, X_original, settings, mask=None):
        '''
        X_orginal = Full (train/valid) features (needed for sampling/drawing)
        settings = dictionary of settings (see default settings)

        '''
        # overwrite keys provided on default settings
        settings = {**default_settings, **settings}
        # print(settings)
        self.method = settings['method']
        self.corruption_rate = settings['corruption_rate']
        self.X_original = X_original
        self.missing = settings['missing']
        self.mask = mask
        # print(self.method, self.mask)
        sampler_map = {
            'mnar': mnar_sampling,
            'mcar': mcar_sampling,
            'mar': mar_sampling,
        }
        self.missing_type = settings['missing_type']
        self.missing_sampler = sampler_map[self.missing_type]
        self.mice = settings['mice']
        
        
    def _get_mask(self, X):
        '''
        TODO: implement without for-loop
        '''
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        X=X.to(device)
        n,d = X.shape
        # debug_mode and print(X.shape)
        d_corrupt = int(self.corruption_rate * d)
        x = np.zeros((n,d))

        for i in range(n):
            a = np.arange(1,d+1)
            a1 = np.random.permutation(a)
            x[i,:] = a1

        mask = np.where(x<=d_corrupt, 1, 0)

        device = X.device
        mask = torch.from_numpy(mask)
        
        # mask = mask if to(device)<0 else mask.to(to(device))
        mask = mask.to(device)
        # debug_mode and print('mask shape', mask.shape)
        # global mask_arr
        # mask_arr = mask
        
        return mask
    def _get_nan_mask(self, X):
        '''
        TODO: implement without for-loop
        '''
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        X = X.to(device)
        n,d = X.shape
        # debug_mode and print(X.shape)
        d_corrupt = int(self.corruption_rate * d)
        x = np.zeros((n,d))

        for i in range(n):
            a = np.arange(1,d+1)
            a1 = np.random.permutation(a)
            x[i,:] = a1

        mask = np.where(x, 1, 0)

        # to( = X.to(device)
        mask = torch.from_numpy(mask)
        
        # mask = mask if to(device)<0 else mask.to(to(device))
        mask = mask.to(device)
        # debug_mode and print('mask shape', mask.shape)
        
        return mask
    
    def _get_c_mask(self, X):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        X = X.clone().to(device)
        nan_mask = torch.where(torch.isnan(X), torch.tensor(1).to(device), torch.tensor(0).to(device))
        return nan_mask
    
    def _zeros(self, X):
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            X = X.to(device)

            # Create a mask for NaN values
            nan_mask = torch.isnan(X)

            # Fill NaN values with zeros
            filled_X = torch.where(nan_mask, torch.tensor(0.0).to(device), X)

            return filled_X

    
    def _median(self, X):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        X = X.to(device)
        # column_means = torch.mean(X[~torch.isnan(X)], dim=0)
        column_means, _ = torch.median(X[~torch.isnan(X)], dim=0)

        # Create a mask for NaN values
        nan_mask = torch.isnan(X)

        # Fill NaN values with the respective column means
        filled_X = torch.where(nan_mask, column_means, X)

        return filled_X


    def _knn(self, X):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # to(device) = X.to(device)
        X = X.to(device)
            
        _, X_missing = self.missing_sampler(pd.DataFrame(X), self.missing, None)
        
        # KNNImputer removes features that has all missing
        # KNNImputer(keep_empty_features=True) in sklearn 1.2 added a parameter
        # However, we are using sklearn 1.0.x and cant upgrade due to depencency contraints
        # so we have to do it ourselves
        empty_cols = X_missing.columns[X_missing.isna().all(axis=0)].values
        X_missing.loc[:, empty_cols] = 0 # check this
        
        knn_imputer = KNNImputer()
        X_imputed = knn_imputer.fit_transform(X_missing)
        X1 = torch.from_numpy(X_imputed.to_numpy()).to(to(device))
        self.mask = self._get_c_mask(X1)
        return X1, self.mask
    
    def _mcar_missing(self, X):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # X = X.to(device)
        # print('corruptor', self.missing)
        _, X_missing = self.missing_sampler(pd.DataFrame(X), self.missing, None)
        # print('NAN values', X_missing.isna().sum().sum())
        X1 = torch.from_numpy(X_missing.to_numpy())
        self.mask = self._get_c_mask(X1)
        # print(1-self.mask)
        # print(torch.sum(self.mask))
        # self.mask = self._get_missing_mask(X1)
        # print(1-self.mask)
        
        return X1, self.mask
    
    
    def _mice(self, X):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # to(device) = X.to(device)
        
        _, X_missing = self.missing_sampler(pd.DataFrame(X), self.missing, None)
        empty_cols = X_missing.columns[X_missing.isna().all(axis=0)].values
        X_missing.loc[:, empty_cols] = 0
        
        kernel = mg.MultipleImputedKernel(
                X_missing,
                datasets=1,
                save_all_iterations=False,
                mean_match_candidates=0,
                initialization='median'
        )
        kernel.mice(self.mice, 1, n_estimators=1, n_jobs=4)
        X_imputed = kernel.complete_data(0)
        X1 = torch.from_numpy(X_imputed.values).to(to(device))
        return X1
    
    def _draw(self, X0):
        ''' 
        replace c*d random select columns for with another random row
        do this for each rows in X0
        where c=corruption_rate and d=number of features
        and X0 is assumed to be unnormalized
        '''
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        X = torch.clone(X0).to(device)
        # to(device) = X.to(device)
        # mask = self._get_c_mask(X)
        _, mask = self._mcar_missing(X)
        # X1 = X.to(device)
        mask = mask.to(device)
        # select random rows for each row (can have same row idx)
        r = torch.randint(self.X_original.shape[0],(X.shape[0],))
        noise_values = self.X_original[r,:].to(device)

        # return (1-mask)*X + mask*imputted
        real = X.mul(1-mask)
        draws = noise_values.mul(mask)

        # print('DRAW:   ',real+draws)

        return torch.tensor(real + draws), mask

    def _draw_error(self, X0):
        ''' 
        replace c*d random select columns for with another random row
        do this for each rows in X0
        where c=corruption_rate and d=number of features
        and X0 is assumed to be unnormalized
        '''
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        X = torch.clone(X0).to(device)
        # to(device) = X.to(device)
        mask = self._get_c_mask(X)
        
        # select random rows for each row (can have same row idx)
        r = torch.randint(self.X_original.shape[0],(X.shape[0],))
        noise_values = self.X_original[r,:]

        # return (1-mask)*X + mask*imputted
        real = X.mul(1-mask)
        draws = noise_values.mul(mask)

        # print('DRAW:   ',real+draws)

        return real + draws
    def _draw_ichi(self, X):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        imputed_tensor = X.clone().to(device)
        nan_indices = torch.isnan(imputed_tensor)
        random_values = torch.randn_like(imputed_tensor)
        
        # Replace NaN values with random values
        imputed_tensor[nan_indices] = random_values[nan_indices]
        
        # print('Draw:', imputed_tensor)
        # return imputed_tensor.cpu().numpy()
        return imputed_tensor

    def _drawX(self, X):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        imputed_tensor = X.clone().to(device)

        # Get the indices of nan values using the mask
        nan_indices = torch.where(torch.isnan(X), torch.tensor(1), torch.tensor(0)).bool()

        # Perform random draw imputation
        for indices in zip(*nan_indices):
            # Get random indices from the original data shape
            random_indices = tuple(torch.randint(imputed_tensor.size(dim), (1,)) for dim in imputed_tensor.shape)

            # Replace the nan values with random values from the original data
            imputed_tensor[indices] = imputed_tensor[random_indices]

        print("Draw: ", imputed_tensor)

        return imputed_tensor
    
    def _noise(self, X0, mean=0, std=1):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        ''' 
        add gaussian noise, N(mu, std) to c*d random columns for all rows
        where c=corruption_rate and d=number of features    
        '''
        X = torch.clone(X0).to(device)

        if mean==0 and std==0: return X

        # to(device) = X.to(device)
        mask = self._get_mask(X)

        noise_values = torch.empty_like(X).normal_(mean, std)
        # noise_values = noise_values if to(device)<0 else noise_values.to(to(device))
        noise_values = noise_values.to(to(device))

        # debug_mode and print(noise_values.shape)
        noise = noise_values.mul(mask)

        return X+noise
    

    def _nanstd(self, x, mean): 
        
        epsilon = 1e-8
        return torch.sqrt(torch.nanmean(torch.pow(x - mean, 2) + epsilon, dim=-1))
    
    def _sample_old(self, X0):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        ''' 
        replace c*d random columns for all rows using original feature distribution
        where c=corruption_rate and d=number of features
        and X0 is assumed to be unnormalized
        '''
        X = torch.clone(X0).to(device)
        # to(device) = X.device

        nan_indices = torch.isnan(X)
        mask = self._get_c_mask(X)
        masked_tensor = self.X_original.masked_select(~nan_indices)

        means = torch.mean(masked_tensor, dim=0)
        stdevs = torch.std(masked_tensor, dim=0)
        # stdevs = self._nanstd(X, means)
        print("MEANS:   ",means)
        
        noise_values = torch.cat([
            torch.empty_like(X[:,i]).normal_(m.item(), s.item()) 
            for i,(m,s) in enumerate(zip(means, stdevs))
        ], dim=-1)

        noise_values = noise_values.reshape(X.shape).contiguous()

        # noise_values = noise_values if device<0 else noise_values.to(device)
        noise_values = noise_values.to(device)

        # return (1-mask)*X + mask*imputted
        real = X.mul(1-mask)
        imputed = noise_values.mul(mask)

        print("sample:   ", real + imputed)

        return real + imputed
    

    def _sample(self, X0):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Get the shape of the input tensor
        X = torch.clone(X0).to(device)
        shape = X.shape
        mask = self.mask.bool()
        noise_values = torch.empty_like(X).normal_()

        # Apply the nan_mask to select the noise values where NaN values are present
        imputed_values = torch.where(mask.to(device), noise_values.to(device), self.X_original.to(device))
        # print("sample:   ",imputed_values)
        return imputed_values
    
    def __call__(self, X):
        
        method_map = {
            'pass': lambda x: x,
            'noise': self._noise,
            'sample': self._sample,
            'draw': self._draw,
            'knn': self._knn,
            'mice': self._mice,
            'mcar_missing' : self._mcar_missing,
            'median': self._median,
            'zeros': self._zeros
        }
        
        return method_map[self.method](X)
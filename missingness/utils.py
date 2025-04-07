# Necessary packages
import numpy as np
import pandas as pd

# PyMC3 for Bayesian Inference
# import pymc3 as pm
from sklearn.preprocessing import StandardScaler

def binary_sampler(p, rows, cols):
    '''Sample binary random variables.

    Args:
    - p: probability of 1
    - rows: the number of rows
    - cols: the number of columns

    Returns:
    - binary_random_matrix: generated binary random matrix.
    '''
    unif_random_matrix = np.random.uniform(0., 1., size=[rows, cols])
    binary_random_matrix = 1 * (unif_random_matrix < p)
    return binary_random_matrix.astype('float32')


def sample_batch_index(total, batch_size):
    '''Sample index of the mini-batch.

    Args:
    - total: total number of samples
    - batch_size: batch size

    Returns:
    - batch_idx: batch index
    '''
    total_idx = np.random.permutation(total)
    batch_idx = total_idx[:batch_size]
    return batch_idx

# def bootstrap_sampling(total, batch_size, seed):
#   '''Sample index using bootstrap sampling.
  
#   Args:
#     - total: total number of samples
#     - batch_size: batch size
#     - seed - seed for np random
    
#   Returns:
#     - sample_idx: sampled batch index
#   '''
#     np.random.seed(seed)
#     arr = range(1,total)
#     sample_idx = np.random.choice(arr, batch_size)
#     return sample_idx


# def mcmc_sampling(X_train, y_column_name, batch_size):
#     with pm.Model() as normal_model:
#         formula = y_column_name + ' ~ ' + ' + '.join(['%s' % variable for variable in X_train.columns[1:]])
#         family = pm.glm.families.Normal()
#         # Creating the model requires a formula and data (and optionally a family)
#         xy = pm.GLM.from_formula(formula, data = X_train, family = family)

#         # Perform Markov Chain Monte Carlo sampling
#         normal_trace = pm.sample(draws=batch_size, chains = 2, tune = 300, cores=-1)
#         return normal_trace

# def bayesian_smapling(data):
#     alphas = np.array([1, 1, 1])
#     # c = np.array([3, 2, 1])
#     c = data

#     # Create model
#     with pm.Model() as model:
#         # Parameters of the Multinomial are from a Dirichlet
#         parameters = pm.Dirichlet('parameters', a=alphas, shape=3)

#         # Observed data is from a Multinomial distribution
#         observed_data = pm.Multinomial('observed_data', n=6, p=parameters, shape=3, observed=c)

#         # Sample from the posterior
#         trace = pm.sample(draws=1000, chains=2, tune=500, discard_tuned_samples=True)


# # def standardize_dataset(dataset, y_column_name):
# #   y_full = dataset[y_column_name].to_numpy()
# #   dataset = dataset.drop([y_column_name], axis=1)
# #   X_full = dataset.to_numpy()
# #   X_standardized = preprocessing.scale(X_full)
# #   standardized_dataset = pd.DataFrame(
# #         data=X_standardized[0:,0:],
# #         index=[i for i in range(X_full.shape[0])],
# #         columns=dataset.columns
# #         )
# #   standardized_dataset[y_column_name] = pd.Series(y_full)
# #   return standardized_dataset 

# def standardize_dataset(dataset, y_column_name):
#     y_full = dataset[y_column_name].to_numpy()
#     dataset = dataset.drop([y_column_name], axis=1)
#     X_full = dataset.to_numpy()
#     scaler = StandardScaler()
#     X_standardized = scaler.fit_transform(X_full)
#     standardized_dataset = pd.DataFrame(
#         data=X_standardized[0:,0:],
#         index=[i for i in range(X_full.shape[0])],
#         columns=dataset.columns
#         )
#     standardized_dataset[y_column_name] = pd.Series(y_full)
#     return standardized_dataset, scaler

# def destandardize_dataset(dataset, y_column_name, scaler):
  
#     if y_column_name is not None: 
#         y_full = dataset[y_column_name].to_numpy()
#         dataset = dataset.drop([y_column_name], axis=1)
#     X_full = dataset.to_numpy()
#     X_standardized = scaler.inverse_transform(X_full)
#     destandardized_dataset = pd.DataFrame(
#         data=X_standardized[0:,0:],
#         index=[i for i in range(X_full.shape[0])],
#         columns=dataset.columns
#         )
#     if y_column_name is not None:
#         destandardized_dataset[y_column_name] = pd.Series(y_full)
# return destandardized_dataset
  


# def poolingMean(df1, df2, df3, df4):
#     return pd.concat([df1, df2, df3, df4]).groupby(level=0).mean()

# def poolingMedian(df1, df2, df3, df4):
#     return pd.concat([df1, df2, df3, df4]).groupby(level=0).median()
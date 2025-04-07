import numpy as np
import pandas as pd
import math
import random

from missingness.utils import sample_batch_index, binary_sampler

def random_sampling(dataframe, no_of_samples):
    no, dim = dataframe.shape

    if no < no_of_samples:
        no_of_samples = no

    data_x = dataframe.values.astype(np.float32)
    sample_idx = sample_batch_index(no, no_of_samples)
    data_x_i = data_x[sample_idx, :]

    actual_dataframe = pd.DataFrame(
        data=data_x_i[0:,0:],
        index=[i for i in range(data_x_i.shape[0])],
        columns=dataframe.columns
        )
    return actual_dataframe

def mcar_sampling(dataframe, miss_rate, no_of_samples):
    '''introduce miss_rate percentage of missing data in a dataset in completely randomly
    Args:
    - data: original data
    - missing_rate: percentage of data missing
    - no_of_samples: no of rows to be samples
    Returns:
    - miss_data_x: dataset with missing data
    '''
    if no_of_samples != None:
        no, dim = dataframe.shape

        if no < no_of_samples:
            no_of_samples = no

        data_x = dataframe.values.astype(np.float32)

        sample_idx = sample_batch_index(no, no_of_samples)
        data_x_i = data_x[sample_idx, :]
    else:
        data_x_i = dataframe.values.astype(np.float32)

    no_i, dim_i = data_x_i.shape

    actual_dataframe = pd.DataFrame(
        data=data_x_i[0:,0:],
        index=[i for i in range(data_x_i.shape[0])],
        columns=dataframe.columns
        )

    # Introduce missing data
    data_m = binary_sampler(1 - miss_rate, no_i, dim_i)
    miss_data_x = data_x_i.copy()
    miss_data_x[data_m == 0] = np.nan


    missing_dataframe = pd.DataFrame(
        data=miss_data_x[0:,0:],
        index=[i for i in range(miss_data_x.shape[0])],
        columns=dataframe.columns
        )

    return actual_dataframe, missing_dataframe

def mar_sampling(dataframe, miss_rate, no_of_samples=None):
    """
    Introduce missing values in a dataset following the MAR (Missing At Random) mechanism.
    
    Args:
    - dataframe (pd.DataFrame): Original dataset.
    - miss_rate (float): Percentage of missing data (e.g., 50% should be sent as 0.5).
    - no_of_samples (int, optional): Number of rows to sample. Defaults to using the entire dataset.

    Returns:
    - actual_dataframe (pd.DataFrame): Dataset before missing values were introduced.
    - missing_dataframe (pd.DataFrame): Dataset with missing values.
    """
    
    if no_of_samples is not None:
        no, dim = dataframe.shape
        if no < no_of_samples:
            no_of_samples = no
        data_x = dataframe.values.astype(np.float32)
        sample_idx = np.random.choice(no, no_of_samples, replace=False)  # Sample without replacement
        data_x_i = data_x[sample_idx, :]
    else:
        data_x_i = dataframe.values.astype(np.float32)

    no_i, dim_i = data_x_i.shape

    # Convert to DataFrame
    actual_dataframe = pd.DataFrame(data=data_x_i, columns=dataframe.columns)

    # Ensure the dataset has enough features
    if dim_i < 2:
        raise ValueError("There should be more than one feature.")
    
    if miss_rate >= 1.0:
        raise ValueError("Miss rate cannot be 100% or more.")

    max_missing = int(no_i * dim_i * miss_rate)
    missing = 0

    quantile_low = miss_rate / 2
    quantile_high = 1 - (miss_rate / 2)

    data_m = np.ones((no_i, dim_i))  # Initialize missing mask

    for i in range(dim_i):
        np.random.seed(i)
        
        # Dynamically select more influencing columns if miss_rate > 90%
        if miss_rate > 0.90:
            num_influencing_cols = min(dim_i - 1, 5)  # Use up to 5 influencing features
        else:
            num_influencing_cols = 3  # Default: 3 influencing features

        # Select influencing columns
        available_columns = list(range(dim_i))
        available_columns.remove(i)
        influencing_cols = np.random.choice(available_columns, num_influencing_cols, replace=False)

        result_indexes = set()
        for sc in influencing_cols:
            high_quantile_mask = actual_dataframe.iloc[:, sc] >= actual_dataframe.iloc[:, sc].quantile(quantile_high)
            low_quantile_mask = actual_dataframe.iloc[:, sc] <= actual_dataframe.iloc[:, sc].quantile(quantile_low)
            result_indexes.update(actual_dataframe[high_quantile_mask | low_quantile_mask].index)

        result_indexes = list(result_indexes)
        random.shuffle(result_indexes)

        column_limit = max(1, int(no_i * miss_rate))  # Ensure at least one value remains
        column_missing = 0

        for j in result_indexes:
            if missing < max_missing and column_missing < column_limit:
                data_m[j, i] = 0  # Set as missing
                column_missing += 1
                missing += 1

    # Apply missing data mask
    miss_data_x = data_x_i.copy()
    miss_data_x[data_m == 0] = np.nan

    # Convert back to DataFrame
    missing_dataframe = pd.DataFrame(data=miss_data_x, columns=dataframe.columns)

    return actual_dataframe, missing_dataframe

# def mar_sampling(dataframe, miss_rate, no_of_samples):
#     '''introduce miss_rate percentage of missing data in a dataset in randomly
#     Args:
#     - data: original data
#     - missing_rate: percentage of data missing (50% should be sent as .5)
#     - no_of_samples: no of rows to be samples
#     Returns:
#     - miss_data_x: dataset with missing data
#     '''

#     if no_of_samples != None:
#         no, dim = dataframe.shape

#         if no < no_of_samples:
#             no_of_samples = no

#         data_x = dataframe.values.astype(np.float32)

#         sample_idx = sample_batch_index(no, no_of_samples)
#         data_x_i = data_x[sample_idx, :]
#     else:
#         data_x_i = dataframe.values.astype(np.float32)
#     no_i, dim_i = data_x_i.shape

#     actual_dataframe = pd.DataFrame(
#         data=data_x_i[0:,0:],
#         index=[i for i in range(data_x_i.shape[0])],
#         columns=dataframe.columns
#         )

#     missing=0
#     j_size = len(data_x_i)
#     max_missing = j_size * len(data_x_i[0]) * miss_rate
    
#     # if dim_i < 5:
#     #     raise ValueError("There should be more than five features")
#     if miss_rate>.85:
#         raise ValueError("Miss rate can not be more than 85 percent")

#     quantile_low = miss_rate / 2
#     quantile_high = 1 - miss_rate / 2

#     for i in range(0, dim_i):
#         np.random.seed(i)
#         if dim_i < 5:
#             available_columns = list(range(dim_i))  # Use all indices
#             available_columns.remove(i)  # Remove the current column index

#             if len(available_columns) < 3:
#                 raise ValueError("Not enough features to select three distinct influencing columns")

#             random.shuffle(available_columns)
#             sc1, sc2, sc3 = available_columns[:3]
#             dfs = []
#             for sc in [sc1, sc2, sc3]:
#                 dfs.append(actual_dataframe[actual_dataframe.columns[sc]] >= actual_dataframe[actual_dataframe.columns[sc]].quantile(quantile_high))
#                 dfs.append(actual_dataframe[actual_dataframe.columns[sc]] <= actual_dataframe[actual_dataframe.columns[sc]].quantile(quantile_low))

#             result_indexes = set().union(*(df.index for df in dfs))
#             result_indexes = list(result_indexes)
#             random.shuffle(result_indexes)
#         else:
#             sc1 = np.random.choice([x for x in range(0,dim_i-1) if x not in [i]])
#             sc2 = np.random.choice([x for x in range(0,dim_i-1) if x not in [i,sc1]])
#             sc3 = np.random.choice([x for x in range(0,dim_i-1) if x not in [i,sc1,sc2]])
#             df_1 = actual_dataframe[actual_dataframe[dataframe.columns[sc1]] >= actual_dataframe[dataframe.columns[sc1]].quantile(quantile_high)]
#             df_2 = actual_dataframe[actual_dataframe[dataframe.columns[sc1]] >= actual_dataframe[dataframe.columns[sc1]].quantile(quantile_low)]
#             df_3 = actual_dataframe[actual_dataframe[dataframe.columns[sc2]] >= actual_dataframe[dataframe.columns[sc2]].quantile(quantile_high)]
#             df_4 = actual_dataframe[actual_dataframe[dataframe.columns[sc2]] >= actual_dataframe[dataframe.columns[sc2]].quantile(quantile_low)]
#             df_5 = actual_dataframe[actual_dataframe[dataframe.columns[sc3]] >= actual_dataframe[dataframe.columns[sc3]].quantile(quantile_high)]
#             df_6 = actual_dataframe[actual_dataframe[dataframe.columns[sc3]] >= actual_dataframe[dataframe.columns[sc3]].quantile(quantile_low)]
#             result_indexes = list(set(df_1.index)|set(df_2.index)|set(df_3.index)|set(df_4.index)|set(df_5.index)|set(df_6.index))
#             random.shuffle(result_indexes)
        
#         data_m_bin = binary_sampler(1, no_i, 1)
#         column_limit = math.ceil(no_i*miss_rate)
#         column_missing = 0
        
#         for j in result_indexes:
#             if missing<max_missing and column_missing<column_limit:
#                 data_m_bin[j] = 0
#                 column_missing+=1
#                 missing+=1
            
#         if 'data_m' in vars():
#             data_m = np.append(data_m, data_m_bin, 1)
#         else:
#             data_m = data_m_bin


    
#     # print("max missing: "+str(max_missing)+":::   total removed:"+str(missing))
#     miss_data_x = data_x_i.copy()
#     miss_data_x[data_m == 0] = np.nan


#     missing_dataframe = pd.DataFrame(
#         data=miss_data_x[0:,0:],
#         index=[i for i in range(miss_data_x.shape[0])],
#         columns=dataframe.columns
#         )

#     return actual_dataframe, missing_dataframe


# def mnar_sampling(dataframe, miss_rate, no_of_samples):
#     '''introduce miss_rate percentage of missing data in a dataset in randomly
#     Args:
#     - data: original data
#     - missing_rate: percentage of data missing (50% should be sent as .5)
#     - no_of_samples: no of rows to be samples
#     Returns:
#     - miss_data_x: dataset with missing data
#     '''

#     if no_of_samples != None:
#         no, dim = dataframe.shape

#         if no < no_of_samples:
#             no_of_samples = no

#         data_x = dataframe.values.astype(np.float32)

#         sample_idx = sample_batch_index(no, no_of_samples)
#         data_x_i = data_x[sample_idx, :]
#     else:
#         data_x_i = dataframe.values.astype(np.float32)

#     no_i, dim_i = data_x_i.shape

#     actual_dataframe = pd.DataFrame(
#         data=data_x_i[0:,0:],
#         index=[i for i in range(data_x_i.shape[0])],
#         columns=dataframe.columns
#         )

#     missing=0
#     j_size = len(data_x_i)
#     max_missing = j_size * len(data_x_i[0]) * miss_rate
#     maxReached = False

#     high = True
#     low = True

#     if dim_i < 2:
#         raise ValueError("There should be more than one feature")
#     if miss_rate>.85:
#         raise ValueError("Miss rate can not be more than 85 percent")

#     quantile_low = miss_rate / 2
#     quantile_high = 1 - (miss_rate / 2)

#     column_limit = math.ceil(no_i*miss_rate)
#     for i in range(0, dim_i):
#         column_missing = 0
#         percentile_high = actual_dataframe[dataframe.columns[i]].quantile(quantile_high)
#         percentile_low = actual_dataframe[dataframe.columns[i]].quantile(quantile_low)
#         data_m_bin = binary_sampler(1, no_i, 1)
#         for j in range (0, no_i):
#             if high and percentile_high <= data_x_i[j][i] and not maxReached and column_missing<column_limit:
#                 data_m_bin[j] = 0
#                 missing+=1
#                 column_missing+=1
#                 if missing >= max_missing:
#                     maxReached = True
#             elif low and percentile_low >= data_x_i[j][i] and not maxReached and column_missing<column_limit:
#                 data_m_bin[j] = 0
#                 missing+=1
#                 column_missing+=1
#                 if missing >= max_missing:
#                     maxReached = True
            
#         if 'data_m' in vars():
#             data_m = np.append(data_m, data_m_bin, 1)
#         else:
#             data_m = data_m_bin

    
#     # print("max missing: "+str(max_missing)+":::   total removed:"+str(missing))
#     miss_data_x = data_x_i.copy()
#     miss_data_x[data_m == 0] = np.nan


#     missing_dataframe = pd.DataFrame(
#         data=miss_data_x[0:,0:],
#         index=[i for i in range(miss_data_x.shape[0])],
#         columns=dataframe.columns
#         )

#     return actual_dataframe, missing_dataframe
def mnar_sampling(dataframe, miss_rate, no_of_samples=None):
    """
    Introduce missing values in a dataset following the MNAR (Missing Not At Random) mechanism.

    Args:
    - dataframe (pd.DataFrame): Original dataset.
    - miss_rate (float): Percentage of missing data (e.g., 50% should be sent as 0.5).
    - no_of_samples (int, optional): Number of rows to sample. Defaults to using the entire dataset.

    Returns:
    - actual_dataframe (pd.DataFrame): Dataset before missing values were introduced.
    - missing_dataframe (pd.DataFrame): Dataset with missing values.
    """

    if no_of_samples is not None:
        no, dim = dataframe.shape
        if no < no_of_samples:
            no_of_samples = no
        data_x = dataframe.values.astype(np.float32)
        sample_idx = sample_batch_index(no, no_of_samples)
        data_x_i = data_x[sample_idx, :]
    else:
        data_x_i = dataframe.values.astype(np.float32)

    no_i, dim_i = data_x_i.shape

    # Convert back to DataFrame
    actual_dataframe = pd.DataFrame(data=data_x_i, columns=dataframe.columns)

    if dim_i < 2:
        raise ValueError("There should be more than one feature.")
    if miss_rate >= 1.0:
        raise ValueError("Miss rate cannot be 100% or more, as it would remove all data.")

    # Initialize missing data mask
    data_m = np.ones((no_i, dim_i))  

    # Set missing rate thresholds
    quantile_low = miss_rate / 2
    quantile_high = 1 - (miss_rate / 2)

    max_missing = int(no_i * dim_i * miss_rate)
    missing = 0
    maxReached = False

    column_limit = math.ceil(no_i * miss_rate)  # Ensure at least some values remain

    for i in range(dim_i):
        column_missing = 0
        percentile_high = actual_dataframe.iloc[:, i].quantile(quantile_high)
        percentile_low = actual_dataframe.iloc[:, i].quantile(quantile_low)

        for j in range(no_i):
            if (data_x_i[j][i] >= percentile_high or data_x_i[j][i] <= percentile_low) and not maxReached:
                if column_missing < column_limit - 1:  # Ensure at least 1 value remains
                    data_m[j, i] = 0  # Mark as missing
                    missing += 1
                    column_missing += 1
                    if missing >= max_missing:
                        maxReached = True

    # Create missing data
    miss_data_x = data_x_i.copy()
    miss_data_x[data_m == 0] = np.nan

    # Convert back to DataFrame
    missing_dataframe = pd.DataFrame(data=miss_data_x, columns=dataframe.columns)

    return actual_dataframe, missing_dataframe
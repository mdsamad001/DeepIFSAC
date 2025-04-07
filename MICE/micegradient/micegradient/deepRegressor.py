# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.wrappers.scikit_learn import KerasRegressor
# from keras.callbacks import EarlyStopping

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf

class DeepRegressor():


    # global log_counter
    # log_counter = 0
    
    def __init__(self):

        # initialize SVD parameters and object
        self.n_components = 5
        self.n_iter = 5
        self.random_state = 42

    def baseline_model(n_1= 256, n_2 = 128, n_3= 64, n_4=32, drop_rate =0.0, x_dimension=0):
        
        n = x_dimension % 4
        n_1 = (x_dimension - n) / 2
        n_2 = n_1 / 2
        n_3 = n_2 / 2

        # create model
        model = Sequential()
        model.add(Dense(n_1, input_dim=x_dimension, activation='relu'))
        # model.add(Dropout(drop_rate))
        model.add(Dense(n_2, activation='relu'))
        # model.add(Dropout(drop_rate))
        model.add(Dense(n_3, activation='relu'))
        # model.add(Dropout(drop_rate))
        #model.add(Dense(n_4, activation='relu'))
    
        model.add(Dense(1))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['mae'])
    
        return model

    def fit(self, X = None, y = None):

        # global log_counter
        # log_counter += 1

        # raise error if X & y parameter is missing
        if X is None or y is None:
            raise ValueError("X & y are required parameters")

        # initialize deep regressor object here to set layers based on data
        self.regressor = self.baseline_model(x_dimension=int(X.shape[1]))
        # csv_logger = keras.callbacks.CSVLogger('log/log'+str(log_counter)+'.csv', append=True, separator=';')
        # es = EarlyStopping(monitor='mae',
        #                       min_delta=0,
        #                       patience=2,
        #                       verbose=0, mode='auto')

        # return self.regressor.fit(X, y, verbose=0, epochs=10, callbacks=[es])
        # return self.regressor.fit(X, y, verbose=0, callbacks=[csv_logger])
        return self.regressor.fit(X, y, verbose=0, epochs=10)

    def predict(self, X = None):

        # raise error if X parameter is missing
        if X is None:
            raise ValueError("X is a required parameter")

        return self.regressor.predict(X)

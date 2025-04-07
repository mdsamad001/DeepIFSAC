from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression

import numpy as np

class LinearRegressionWithSVD():
    
    def __init__(self):

        # initialize SVD parameters and object
        self.n_components = 5
        self.n_iter = 5
        self.random_state = 42

        self.svd = TruncatedSVD(n_components=self.n_components, n_iter=self.n_iter, random_state=self.random_state)
        
        # initialize linear regressor object and return
        self.regressor = LinearRegression()

    def fit(self, X = None, y = None):

        # raise error if X & y parameter is missing
        if X is None or y is None:
            raise ValueError("X & y are required parameters")

        # transform X using svd before feeding into linear regressor    
        X_svd = self.svd.fit_transform(X)

        # initialize weights randomly between 0 and 1
        # W_init = np.random.uniform(X_svd.shape[1])

        return self.regressor.fit(X_svd, y)

        # print(self.regressor.coef_)
        # print(self.regressor.intercept_)

    def predict(self, X = None):

        # raise error if X parameter is missing
        if X is None:
            raise ValueError("X is a required parameter")

        # transform X using svd before feeding into linear regressor    
        X_svd = self.svd.fit_transform(X)


        return self.regressor.predict(X_svd)
        


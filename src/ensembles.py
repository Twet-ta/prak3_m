import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import timeit
from scipy.optimize import minimize_scalar
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import random
import matplotlib
import matplotlib.pyplot as plt

import scipy
import matplotlib.ticker
import seaborn as sns
import pickle


class RandomForestMSE:
    def __init__(self, n_estimators, max_depth=None, feature_subsample_size=None, **trees_parameters):
        self.n_est = n_estimators
        self.depth = max_depth
        self.f_size = feature_subsample_size
        self.clf = []
        self.clf_f = []
        self.par = trees_parameters
        """
        n_estimators : int
            The number of trees in the forest.
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """

    def fit(self, X, y, X_val=None, y_val=None):
        if self.f_size is None:
            self.f_size = X.shape[1] // 3
        if not (X_val is None):
            history = {}
            history['acc'] = []
            history['time'] = []
            history['time'].append(0)
            history['acc'].append(0)
            sum = np.zeros(X_val.shape[0])
        for i in range(self.n_est):
            a = timeit.default_timer()
            arr = np.random.choice(X.shape[1], size=self.f_size, replace=True, p=None)
            tmp_clf = DecisionTreeRegressor(**self.par, max_depth=self.depth)
            tmp_clf.fit(X[:, arr], y)
            self.clf.append(tmp_clf)
            self.clf_f.append(arr)
            if not (X_val is None):
                sum = sum + tmp_clf.predict(X_val[:, arr])
                t = history['time'][-1]
                history['time'].append(timeit.default_timer() - a + t)
                history['acc'].append(mean_squared_error(y_val, sum / self.n_est, squared=False))
        if not (X_val is None):
            return history
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        X_val : numpy ndarray
            Array of size n_val_objects, n_features
        y_val : numpy ndarray
            Array of size n_val_objects
        """

    def predict(self, X):
        sum = np.array(X.shape[0])
        for i in range(self.n_est):
            sum = sum + self.clf[i].predict(X[:, self.clf_f[i]])
        return sum / self.n_est
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """


class GradientBoostingMSE:
    def __init__(self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None, **trees_parameters):
        self.n_est = n_estimators
        self.depth = max_depth
        self.f_size = feature_subsample_size
        self.par = trees_parameters
        self.lr = learning_rate
        self.clf = []
        self.clf_f = []

        """
        n_estimators : int
            The number of trees in the forest.
        learning_rate : float
            Use alpha * learning_rate instead of alpha
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """

    def fit(self, X, y, X_val=None, y_val=None):
        if self.f_size is None:
            self.f_size = X.shape[1] // 3
        if not (X_val is None):
            history = {}
            history['acc'] = []
            history['time'] = []
            history['time'].append(0)
            history['acc'].append(0)
            sum = np.zeros(X_val.shape[0])
        b = np.zeros(self.n_est)
        a = timeit.default_timer()
        arr = np.random.choice(X.shape[1], size=self.f_size, replace=True, p=None)
        tmp_clf = DecisionTreeRegressor(**self.par, max_depth=self.depth)
        tmp_clf.fit(X[:, arr], y)
        self.clf_f.append(arr)
        self.clf.append(tmp_clf)
        if not (X_val is None):
            t = history['time'][-1]
            history['time'].append(timeit.default_timer() - a + t)
            history['acc'].append(mean_squared_error(y_val, sum, squared=False))
        m = np.zeros(X.shape[0])
        for i in range(1, self.n_est):
            a = timeit.default_timer()
            arr = np.random.choice(X.shape[1], size=self.f_size, replace=True, p=None)
            tmp_clf = DecisionTreeRegressor(**self.par, max_depth=self.depth)
            m = m + self.lr * b[i - 1] * self.clf[i - 1].predict(X[:, self.clf_f[i - 1]])
            tmp_clf.fit(X[:, arr], 2 * (y - m))

            def f(k):
                sum = m + k * tmp_clf.predict(X[:, arr]) - y
                return np.sum(sum ** 2)

            tmp = minimize_scalar(f)
            b[i] = tmp.x
            self.clf.append(tmp_clf)
            self.clf_f.append(arr)
            if not (X_val is None):
                sum = sum + self.lr * b[i] * tmp_clf.predict(X_val[:, arr])
                t = history['time'][-1]
                history['time'].append(timeit.default_timer() - a + t)
                history['acc'].append(mean_squared_error(y_val, sum, squared=False))
        self.b = b
        return history
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        """

    def predict(self, X):
        sum = np.array(X.shape[0])
        for i in range(self.n_est):
            sum = sum + self.lr * self.b[i] * self.clf[i].predict(X[:, self.clf_f[i]])
        return sum
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
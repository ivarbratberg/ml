import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from skopt import gp_minimize
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
from bayesian_optimization_util import plot_approximation, plot_acquisition
from sklearn import datasets
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from scipy.stats import uniform
from xgboost import XGBRegressor
# import GPy
# import GPyOpt
# from GPyOpt.methods import BayesianOptimization
import sys

def bayesian(X,Y):
    bds = [{'name': 'learning_rate', 'type': 'continuous', 'domain': (0, 1)},
            {'name': 'gamma', 'type': 'continuous', 'domain': (0, 5)},
            {'name': 'max_depth', 'type': 'discrete', 'domain': (1, 50)},
            {'name': 'n_estimators', 'type': 'discrete', 'domain': (1, 300)},
            {'name': 'min_child_weight', 'type': 'continuous', 'domain': (1, 100)},
            {'name': 'colsample_bytree', 'type': 'continuous', 'domain': (0.1, 0.8)},
            {'name': 'subsample', 'type': 'continuous', 'domain': (0.1, 0.8)}  
            ]


    noise = 0
    m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    gpr = GaussianProcessRegressor(kernel=m52, alpha=noise**2)
        

    # Optimization objective
    def cv_score(parameters):
        # parameters = parameters[0]
        score = cross_val_score(
                    XGBRegressor(learning_rate=parameters[0],
                                gamma=int(parameters[1]),
                                max_depth=int(parameters[2]),
                                n_estimators=int(parameters[3]),
                                min_child_weight = parameters[4],
                                colsample_bytree = parameters[5],
                                subsample = parameters[6]),
                    X, Y, scoring='neg_mean_squared_error').mean()
        print(score)
        return score

    # optimizer = BayesianOptimization(f=cv_score, 
    #                                 domain=bds,
    #                                 model_type='GP',
    #                                 acquisition_type ='EI',
    #                                 acquisition_jitter = 0.05,
    #                                 exact_feval=True, 
    #                                 maximize=True)

    # On|ly 20 iterations because we have 5 initial random points
    opti_obj = gp_minimize(lambda x: cv_score(x), 
                        map(lambda x:x['domain'],bds),
                        base_estimator=gpr,
                        acq_func='EI',      # expected improvement
                        xi=0.01,            # exploitation-exploration trade-off
                        n_calls=10,         # number of iterations
                        n_random_starts=4  # initial samples are provided
    )
    return opti_obj

        
    
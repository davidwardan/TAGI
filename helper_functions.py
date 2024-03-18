# Import libraries
import math

import numpy as np
from numpy import random

# define the function to generate data for y=x3 + v
def generate_data(n, obs_mean, obs_var):
    # generate random x data
    x = np.linspace(-5, 5, n)
    # generate y
    y = x**3 + random.normal(obs_mean, obs_var, n)

    return x, y

# define the relu activation function
def relu(x):
    return np.maximum(0, x)

# define the derivative of the relu activation function where input is a matrix
def relu_derivative(matrix):
    return np.where(matrix > 0, 1, 0)

# define the observation model
def observation(mu_z0, var_z0, var_v):
    # get y
    mu_y = mu_z0
    var_y = var_z0 + var_v

    return mu_y, var_y




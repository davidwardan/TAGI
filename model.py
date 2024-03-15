# import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import pandas as pd

# get helper functions
from TAGI import *

# set the random seed
np.random.seed(235)

# generate training data
x_train, y_train = generate_data(20, 0, 9)
# generate test data
x_test, y_test = generate_data(20, 0, 9)

# plot the training data and testing data
# plt.scatter(x_train, y_train, label='Training Data')
# plt.scatter(x_test, y_test, label='Testing Data')
# plt.legend()
# plt.show()

# define the number of layers, number of neurons in each layer
n_layers = 1
n_units = 100
input_dim = 1 # considering that the input is implemented in one batch
output_dim = 1

var_v = 9

# Initialize the weights
sigma_w1 = (1/n_units) * np.eye(n_units)
sigma_w2 = (1/n_units) * np.eye(n_units) # TODO: check dimensions

mu_w1 = np.random.normal(0, (1/n_units), (n_units, input_dim))
mu_w2 = np.random.normal(0, (1/n_units), (output_dim, n_units))

# Initialize the biases
sigma_b1 = (1/n_units) * np.eye(n_units)
sigma_b2 = (1/n_units) * np.ones((output_dim, 1)) #TODO: check dimensions

mu_b1 = np.random.normal(0, (1/n_units), (n_units, input_dim))
mu_b2 = np.random.normal(0, (1/n_units), (output_dim, 1))

# print size of the weights and biases
print('mu_w1:', mu_w1.shape)
print('mu_w2:', mu_w2.shape)
print('mu_b1:', mu_b1.shape)
print('mu_b2:', mu_b2.shape)

# define the number of epochs
n_epochs = 5
log_ll = []

# train the model
for epoch in range(n_epochs):
    log_likelihood = 0
    for x, y in zip(x_train,y_train):
        # aplly the forward pass from input to hidden layer
        mu_z, var_z, cov_z_w, cov_z_b = forward_input(mu_w1, sigma_w1, mu_b1, sigma_b1, x)

        # apply the activation function on the hidden layer
        mu_a, var_a, cov_a_w, cov_a_b, J = activation(mu_z, var_z, cov_z_w, cov_z_b)

        # apply the forward pass from hidden to the output layer
        mu_z0, var_z0, cov_z0_w, cov_z0_b = forward_output(mu_w2, sigma_w2, mu_b2, sigma_b2, mu_a, var_a)

        # apply the observation model
        mu_y, var_y = observation(mu_z0, var_z0, var_v)

        # get the log likelihood
        log_likelihood += np.log(var_y) + (y - mu_y)**2 / var_y #TODO: check the log likelihood

        # aplly first backward step, from y to z0
        mu_z0_post, var_z0_post = update_output(var_y, mu_y, mu_z0, var_z0, y)

        # apply the second backward step, from z0 to z
        mu_post_z, var_post_z = update_hidden(J, var_z, var_z0, mu_z, mu_z0, mu_z0_post, var_z0_post, mu_w2)

        # update the parameters
        (mu_post_w2, var_post_w2, mu_post_b2, var_post_b2,
         mu_post_w1, var_post_w1, mu_post_b1, var_post_b1) = update_parameters(var_z0, cov_z0_w, cov_z0_b, mu_w2, mu_z0_post, mu_z0, sigma_w2, var_z0_post, mu_b2, sigma_b2,
                                                                               cov_z_w, cov_z_b, var_z, mu_w1, mu_post_z, mu_z, var_post_z)

        # update all the parameters in the architecture before the next input
        mu_w2 = mu_post_w2
        sigma_w2 = var_post_w2
        mu_b2 = mu_post_b2
        sigma_b2 = var_post_b2
        mu_w1 = mu_post_w1
        sigma_w1 = var_post_w1
        mu_b1 = mu_post_b1
        sigma_b1 = var_post_b1

    # append the log likelihood
    log_ll.append(log_likelihood/len(x_train))


# plot the log likelihood
plt.plot(log_ll)
plt.title('Log Likelihood')
plt.xlabel('Epoch')
plt.ylabel('Log Likelihood')
plt.show()















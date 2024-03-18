# import libraries
import numpy as np
from helper_functions import *

# Define the funcion to apply first forward pass from input to first layer
def forward_input(mu_w1, sigma_w1, mu_b1, sigma_b1, x_train):
    # Perform the forward pass from input to hidden layer
    mu_z = mu_w1 * x_train + mu_b1
    var_z = sigma_w1 * (x_train ** 2) + sigma_b1  # TODO: fix notation
    cov_z_w = x_train * sigma_w1
    cov_z_b = sigma_b1

    return mu_z, np.diag(np.diag(var_z)), np.diag(np.diag(cov_z_w)), np.diag(np.diag(cov_z_b))

# Apply the activation function on a layer Z
def activation(mu_z, var_z, cov_z_w, cov_z_b):
    J = relu_derivative(mu_z)
    J = np.diag(J.flatten())
    mu_a = J @ mu_z
    var_a = J @ var_z @ J.T
    cov_a_w = J @ cov_z_w
    cov_a_b = J @ cov_z_b

    return mu_a, np.diag(np.diag(var_a)), np.diag(np.diag(cov_a_w)), np.diag(np.diag(cov_a_b)), J

# Apply the forward pass from hidden to the output layer
def forward_output(mu_w2, sigma_w2, mu_b2, sigma_b2, mu_a, var_a):
    f1 = np.ones((1, 100))

    # now perform the forward pass from the hidden layer to the output layer
    mu_z0 = mu_w2 @ mu_a + mu_b2
    var_z0 = f1 @ (sigma_w2 @ var_a + sigma_w2 @ mu_a ** 2 ) @ f1.T + f1 @ (var_a @ mu_w2.T ** 2 ) + sigma_b2
    cov_z0_w = f1 @ sigma_w2 @ mu_a
    cov_z0_b = sigma_b2

    return mu_z0, var_z0, cov_z0_w, cov_z0_b

def observation(mu_z0, var_z0, var_v):
    # get y
    mu_y = mu_z0
    var_y = var_z0 + var_v

    return mu_y, var_y

# Define the backward step from observation to the output layer
def update_output(var_y, mu_y, mu_z0, var_z0, y_train):
    cov_y_z0 = var_z0
    mu_z0_post = mu_z0 + cov_y_z0.T @ np.linalg.inv(var_y) @ (y_train - mu_y)
    var_z0_post = var_z0 - cov_y_z0.T @ np.linalg.inv(var_y) @ cov_y_z0

    return mu_z0_post, var_z0_post

# Define the backward step from the output layer to the hidden layer
def update_hidden(J, var_z, var_z0, mu_z, mu_z0, mu_z0_inf, var_z0_inf, mu_w2):
    # apply one backward pass for TAGI from the output layer to the hidden layer
    cov_zplus_z = var_z @ J.T @ mu_w2.T  # TODO: check formulation
    Jz = cov_zplus_z @ np.linalg.inv(var_z0)
    mu_post_z = mu_z + Jz @ (mu_z0_inf - mu_z0)  # TODO: transpose Jz
    var_post_z = var_z + Jz @ (var_z0_inf - var_z0) @ Jz.T

    return mu_post_z, np.diag(np.diag(var_post_z))

def update_parameters(var_z0, cov_z0_w, cov_z0_b, mu_w2, mu_z0_post, mu_z0, sigma_w2, var_z0_post, mu_b2, sigma_b2, sigma_w1, sigma_b1, mu_b1, cov_z_w, cov_z_b, var_z, mu_w1, mu_post_z, mu_z, var_post_z):
    # apply one backward pass for TAGI from the output layer to the weights and biases
    Jw2 = cov_z0_w @ np.linalg.inv(var_z0)
    Jb2 = cov_z0_b @ np.linalg.inv(var_z0)

    mu_post_w2 = mu_w2 + Jw2 @ (mu_z0_post - mu_z0)
    var_post_w2 = sigma_w2 + Jw2 @ (var_z0_post - var_z0) @ Jw2.T

    mu_post_b2 = mu_b2 + Jb2 @ (mu_z0_post - mu_z0)
    var_post_b2 = sigma_b2 + Jb2 @ (var_z0_post - var_z0) @ Jb2.T

    # apply one backward pass for TAGI from the output layer to the weights and biases
    Jw1 = cov_z_w @ np.linalg.inv(var_z)
    Jb1 = cov_z_b @ np.linalg.inv(var_z)

    mu_post_w1 = mu_w1 + Jw1 @ (mu_post_z - mu_z)
    var_post_w1 = sigma_w1 + Jw1 @ (var_post_z - var_z) @ Jw1.T

    mu_post_b1 = mu_b1 + Jb1 @ (mu_post_z - mu_z)
    var_post_b1 = sigma_b1 + Jb1 @ (var_post_z - var_z) @ Jb1.T

    return mu_post_w2, np.diag(np.diag(var_post_w2)), mu_post_b2, var_post_b2, mu_post_w1, np.diag(np.diag(var_post_w1)), mu_post_b1, np.diag(np.diag(var_post_b1))
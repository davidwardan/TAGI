# import the necessary packages
import matplotlib.pyplot as plt
from tqdm import tqdm

# get helper functions
from TAGI import *

# set the random seed
np.random.seed(415)

# generate training data
x_train, y_train = generate_data(20, 0, 9)
# generate test data
x_test, y_test = generate_data(20, 0, 9)

# Normalize the data between [0,1]
x_train_norm = (x_train - np.mean(x_train)) / np.std(x_train)
y_train_norm = (y_train - np.mean(y_train)) / np.std(y_train)
x_test_norm = (x_test - np.mean(x_train)) / np.std(x_train)
y_test_norm = (y_test - np.mean(y_train)) / np.std(y_train)


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

var_v = 9 / np.var(y_train)  #TODO: check the variance of the noise (9 is the variance of the data)

# Initialize the weights
sigma_w1 = (1/input_dim) * np.eye(n_units)
sigma_w2 = (1/n_units) * np.eye(n_units) # TODO: check dimensions

mu_w1 = np.random.normal(0, 1, (n_units, input_dim))
mu_w2 = np.random.normal(0, (np.sqrt(1/n_units)), (output_dim, n_units))

# Initialize the biases
sigma_b1 = (1/input_dim) * np.eye(n_units)
sigma_b2 = (1/n_units) * np.ones((output_dim, 1)) #TODO: check dimensions

mu_b1 = np.random.normal(0, 1, (n_units, input_dim))
mu_b2 = np.random.normal(0, np.sqrt((1/n_units)), (output_dim, 1))

# print size of the weights and biases
print('mu_w1:', mu_w1.shape)
print('mu_w2:', mu_w2.shape)
print('mu_b1:', mu_b1.shape)
print('mu_b2:', mu_b2.shape)

# define the number of epochs
n_epochs = 20
log_ll_train = []
log_ll_test = []

# train the modela
for epoch in tqdm(range(n_epochs)):
    log_likelihood_train = 0
    for x, y in zip(x_train_norm,y_train_norm):
        # aplly the forward pass from input to hidden layer
        mu_z, var_z, cov_z_w, cov_z_b = forward_input(mu_w1, sigma_w1, mu_b1, sigma_b1, x)

        # apply the activation function on the hidden layer
        mu_a, var_a, cov_a_w, cov_a_b, J = activation(mu_z, var_z, cov_z_w, cov_z_b)

        # apply the forward pass from hidden to the output layer
        mu_z0, var_z0, cov_z0_w, cov_z0_b = forward_output(mu_w2, sigma_w2, mu_b2, sigma_b2, mu_a, var_a)

        # apply the observation model
        mu_y, var_y = observation(mu_z0, var_z0, var_v)

        # get the log likelihood
        log_likelihood_train += -0.5 * np.log(2 * np.pi) - 0.5 * np.log(var_y) - 0.5 * (y - mu_y)**2 / var_y #TODO: correct the log likelihood

        # aplly first backward step, from y to z0
        mu_z0_post, var_z0_post = update_output(var_y, mu_y, mu_z0, var_z0, y)

        # apply the second backward step, from z0 to z
        mu_post_z, var_post_z = update_hidden(J, var_z, var_z0, mu_z, mu_z0, mu_z0_post, var_z0_post, mu_w2) #TODO: use diagonal

        # update the parameters
        (mu_post_w2, var_post_w2, mu_post_b2, var_post_b2,
         mu_post_w1, var_post_w1, mu_post_b1, var_post_b1) = update_parameters(var_z0, cov_z0_w, cov_z0_b, mu_w2, mu_z0_post, mu_z0, sigma_w2, var_z0_post, mu_b2, sigma_b2, sigma_w1, sigma_b1, mu_b1, cov_z_w, cov_z_b, var_z, mu_w1, mu_post_z, mu_z, var_post_z)
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
    log_ll_train.append(log_likelihood_train.flatten()/len(x_train))

    # evaluate the model
    y_pred = []
    var_pred = []
    log_likelihood_test = 0
    for x, y in zip(x_test_norm, y_test_norm):
        # apply the forward pass from input to hidden layer
        mu_z, var_z, cov_z_w, cov_z_b = forward_input(mu_w1, sigma_w1, mu_b1, sigma_b1, x)

        # apply the activation function on the hidden layer
        mu_a, var_a, cov_a_w, cov_a_b, J = activation(mu_z, var_z, cov_z_w, cov_z_b)

        # apply the forward pass from hidden to the output layer
        mu_z0, var_z0, cov_z0_w, cov_z0_b = forward_output(mu_w2, sigma_w2, mu_b2, sigma_b2, mu_a, var_a)

        # apply the observation model
        mu_y, var_y = observation(mu_z0, var_z0, var_v)

        # get the log likelihood
        log_likelihood_test += -0.5 * np.log(2 * np.pi) - 0.5 * np.log(var_y) - 0.5 * (y - mu_y)**2 / var_y

        y_pred.append(mu_y.flatten()[0])
        var_pred.append(var_y.flatten()[0])

    # append the log likelihood
    log_ll_test.append(log_likelihood_test.flatten()/len(x_test))

# check at which epoch the log likelihood for the test data is the highest
print('Epoch with the highest log likelihood:', np.argmax(log_ll_test))


# plot the log likelihood for the training data and the testing data and show on plot the epoch with the highest log likelihood
plt.figure(figsize=(10, 5))
plt.rcParams.update({'font.size': 12})
plt.plot(log_ll_train, label='Training Data', color='steelblue')
plt.plot(log_ll_test, label='Testing Data', color = 'grey')
plt.axvline(np.argmax(log_ll_test), color='red', linestyle='--', label='Best Epoch')
plt.xlabel('Epochs')
plt.ylabel('Log Likelihood')
plt.xticks(np.arange(0, n_epochs, 1))
plt.legend()
plt.show()

# plot the results where the predicted is a line plot and the true data is a scatter plot
plt.figure(figsize=(5, 5))
plt.rcParams.update({'font.size': 12})
plt.scatter(x_test_norm, y_test_norm, label='True Data', color='forestgreen', alpha=0.7, edgecolor='black')
plt.plot(x_test_norm, y_pred, label='Predicted Data', color= 'darkorange')
plt.fill_between(x_test_norm, np.array(y_pred) - np.sqrt(var_pred), np.array(y_pred) + np.sqrt(var_pred), alpha=0.3, label='Uncertainty', color='grey')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


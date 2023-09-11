# from utils import *
import matplotlib.pyplot as plt
import numpy as np
import math
# from test_utils import * 
import importlib.util

def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

module = module_from_file("load_data",
     "/home/drsystem/Documents/search for data science/Coursera_Supervised_Machine_Learning_Regression_and_Classification_2022-6_Downloadly.ir/assignments/logistic_regression/utils.py")

# if __name__ == "__main__":
#     print(foo)
#     print(dir(foo))
#     foo.announce()


def plot_data_set(x_train, y_train):
    positive = y_train == 1
    negative = y_train == 0

    plt.plot(x_train[positive, 0], x_train[positive, 1], 'k+', label='admitted')
    plt.plot(x_train[negative, 0], x_train[negative, 1], 'yo', label='not admitted')
    plt.legend(loc='upper right')

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def compute_loss(xi, yi, w, b):
    z = np.dot(xi, w) + b
    f_wb = sigmoid(z)

    loss = -yi * math.log(f_wb) - (1 - yi) * math.log(1 - f_wb)
    return loss


def compute_cost(x, y, w, b):
    m = x.shape[0]

    cost = 0
    for i in range(m):
        cost += compute_loss(x[i], y[i], w, b)
    
    cost = cost / m

    return cost


def compute_gradients(x, y, w, b):
    m = x.shape[0]
    n = w.shape[0]

    dj_db = 0 
    dj_dwj = np.zeros(n)

    for i in range(m):
        f_wb = np.dot(w, x[i]) + b
        cost = sigmoid(f_wb) - y[i]
        
        dj_db += cost

        for j in range(n):
            dj_dwj[j] += cost * x[i, j]
    
    dj_dwj = dj_dwj / m
    dj_db = dj_db / m

    return dj_dwj, dj_db

def plot_decisionboundary(w, b, x, y):
    x_plot = np.array([min(x[:, 0]), max(x[:, 0])])
    y_plot = -1 / (w[1]) * (b + w[0] * x_plot)

    plt.plot(x_plot, y_plot, c='b')


def gradient_descent(x, y, w, b, learning_rate, iterations):

    m = x.shape[0]
    j_history = []
    
    for i in range(iterations):
        dj_dwj, dj_db = compute_gradients(x, y, w, b)

        b = b - learning_rate * dj_db 
        w = w - learning_rate * dj_dwj

        j_history.append(compute_cost(x, y, w, b))

        if i % 1000 == 0:
            print(f'Iteration {i:4}: Cost {j_history[-1]}')

    return w, b, j_history

def predict(x, w, b):
    m = x.shape[0]

    predictions = np.zeros(m)
    
    for i in range(m):
        z_i = np.dot(x[i], w)
        z_i += b

        f_wb = sigmoid(z_i)

        predictions[i] = 1 if f_wb >= 0.5 else 0

    return predictions

if __name__ == "__main__":

    x_train, y_train = module.load_data('./non_regularized/ex2data1.txt')
    m = y_train.shape[0]
    n = x_train.shape[1]
    print('in logistic regression')
    # print(f'type of x_train is {type(x_train)}')
    # print(f'type of y_train is {type(y_train)}')
    # print(f'dimensions of x_train are {x_train.shape}')
    # print(f'dimensions of y_train are {y_train.shape}')

    # --------------------- test predict ---------------------
    # np.random.seed(1)
    # tmp_w = np.random.randn(2)
    # tmp_b = 0.3    
    # tmp_X = np.random.randn(4, 2) - 0.5

    # tmp_p = predict(tmp_X, tmp_w, tmp_b)
    # print(f'Output of predict: shape {tmp_p.shape}, value {tmp_p}')

    # ----------------- compute the accuracy of model -----------------

    # UNIT TESTS        


    # test 1
    # m, n = x_train.shape

    # Compute and display cost with w initialized to zeroes
    # initial_w = np.zeros(n)
    # initial_b = 0.
    # cost = compute_cost(x_train, y_train, initial_w, initial_b)
    # print('Cost at initial w (zeros): {:.3f}'.format(cost))

    # test 2
    # test_w = np.array([0.2, 0.2])
    # test_b = -24.
    # cost = compute_cost(x_train, y_train, test_w, test_b)

    # print('Cost at test w,b: {:.3f}'.format(cost))

    # ------------------------- test compute_gradient -------------------------
    # test 1
    # initial_w = np.zeros(n)
    # initial_b = 0.

    # dj_dw, dj_db = compute_gradients(x_train, y_train, initial_w, initial_b)
    # print(f'dj_db at initial w (zeros):{dj_db}' )
    # print(f'dj_dw at initial w (zeros):{dj_dw.tolist()}' )
    # test 2
    # test_w = np.array([ 0.2, -0.5])
    # test_b = -24
    # dj_dwj, dj_db  = compute_gradients(x_train, y_train, test_w, test_b)

    # print('dj_db at test_w:', dj_db)
    # print('dj_dw at test_w:', dj_dwj.tolist())

    # ----------------- test gradient descent -----------------
    np.random.seed(1)
    initial_w = 0.01 * (np.random.rand(2).reshape(-1,1) - 0.5).flatten()
    initial_b = -8
    iterations = 10000
    alpha = 0.001
    w, b, _ = gradient_descent(x_train, y_train, initial_w, initial_b, alpha, iterations)

    # ----------------- decision boundary -----------------

    # plot_data_set(x_train, y_train)
    # plot_decisionboundary(w, b, x_train, y_train)

    # plt.show()


    # ------------------------ accuracy of model ------------------------
    predictions = predict(x_train, w, b)
    print(f'accuracy is {np.mean(predictions == y_train) * 100}')

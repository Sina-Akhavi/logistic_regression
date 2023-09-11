import matplotlib.pyplot as plt
import numpy as np
import math 
import importlib.util

def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

utils = module_from_file("utils",
     "/home/drsystem/Documents/search for data science/Coursera_Supervised_Machine_Learning_Regression_and_Classification_2022-6_Downloadly.ir/assignments/logistic_regression/utils.py")
logistic_regression = module_from_file("logisticregression",
     "/home/drsystem/Documents/search for data science/Coursera_Supervised_Machine_Learning_Regression_and_Classification_2022-6_Downloadly.ir/assignments/logistic_regression/non_regularized/logisticregression.py")
test_utils = module_from_file("test_utils",
     "/home/drsystem/Documents/search for data science/Coursera_Supervised_Machine_Learning_Regression_and_Classification_2022-6_Downloadly.ir/assignments/logistic_regression/test_utils.py")

def visualize_dataset(x, y):
    positive = y == 1
    negative = y == 0

    plt.plot(x[positive, 0], x[positive, 1], 'k+', label='accepted')
    plt.plot(x[negative, 0], x[negative, 1], 'yo', label='rejected')
    plt.legend(loc='upper right')

def compute_cost_reg(x, y, w, b, lambda_=1):
    term1 = logistic_regression.compute_cost(x, y, w, b)

    m, n = x.shape

    term2 = (np.dot(w, w) ) * lambda_ / (2 * m)

    cost = term1 + term2
    return cost

def compute_gradients_reg(x, y, w, b, lambda_=1):
    m = x.shape[0]
    dj_dw, dj_db = logistic_regression.compute_gradients(x, y, w, b)
    dj_dw += lambda_ / m * w

    return dj_dw, dj_db

def gradient_descent(x, y, w, b, learning_rate, lambda_, iterations, compute_gradients_reg, 
                     compute_cost_reg):

    j_history = np.zeros(iterations)

    for i in range(iterations):
        dj_dw, dj_db = compute_gradients_reg(x, y, w, b, lambda_)

        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db

        j_history[i] = compute_cost_reg(x, y, w, b, lambda_)

        if i % 1000 == 0:
            print(f'iteration {i:4}  cost:{j_history[i]}')
    
    return w, b, j_history

def predict(w, b, x_mapped, y):
    m = x_mapped.shape[0]
    predictions = np.zeros(m)
    
    for i in range(m):  
        z = np.dot(w, x_mapped[i]) + b

        f_wb = logistic_regression.sigmoid(z)

        predictions[i] = 1 if f_wb >= 0.5 else 0
    
    return predictions

# ---------------- feature map ----------------

x_train, y_train = utils.load_data('./regularized/ex2data2.txt')

x_train_mapped = utils.map_feature(x_train[:, 0], x_train[:, 1])
# print(f'x_train[0] = {x_train[0]}')
# print(f'x_train_mapped[0] = {x_train_mapped[0]}')

np.random.seed(1)
initial_w = np.random.rand(x_train_mapped.shape[1])-0.5
initial_b = 1.

# Set regularization parameter lambda_ to 1 (you can try varying this)
lambda_ = 0.01                                          
# Some gradient descent settings
iterations = 10000
alpha = 0.01

w,b, J_history = gradient_descent(x_train_mapped, y_train, initial_w, initial_b, alpha,
                                    lambda_, iterations, compute_gradients_reg, compute_cost_reg)

utils.plot_decision_boundary(w, b, x_train_mapped, y_train)
plt.show()

# ------------------ Model Evaluation ------------------
predictions = predict(w, b, x_train_mapped, y_train)

print(f'the accuracy of the model is {np.mean(predictions == y_train) * 100}')


# ------------------ check computations of gradients ------------------
# np.random.seed(1) 
# initial_w  = np.random.rand(x_train_mapped.shape[1]) - 0.5 
# initial_b = 0.5
 
# lambda_ = 0.5
# dj_dw, dj_db = compute_gradients_reg(x_train_mapped, y_train, initial_w, initial_b, lambda_)

# print(f"dj_db: {dj_db}", )
# print(f"First few elements of regularized dj_dw:\n {dj_dw[:4].tolist()}", )
# 









# ------------------- test compute_cost_reg correctness -------------------
# np.random.seed(1)
# initial_w = np.random.rand(x_train_mapped.shape[1]) - 0.5
# initial_b = 0.5
# lambda_ = 0.5
# cost = compute_cost_reg(x_train_mapped, y_train, initial_w, initial_b, lambda_)

# print("Regularized cost :", cost)
# 
# test_utils.compute_cost_reg_test(compute_cost_reg) 

# ------------- visualize dataset -------------
# visualize_dataset(x_train, y_train)
# plt.show()
# 

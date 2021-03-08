import math
import statistics
import numpy as np
import matplotlib.pyplot as plot
from numpy.linalg import inv

############################# supporting functions ################################
# Load train.csv and test.csv
with open('train.csv') as f:
    training_data = [];
    for line in f:
        terms = line.strip().split(',')
        training_data.append(terms)

with open('test.csv') as f:
    testing_data = [];
    for line in f:
        terms = line.strip().split(',')
        testing_data.append(terms)

def convert_to_float(input_data):
    for element in input_data:
        for indx in range(len(input_data[0])):
            element[indx] = float(element[indx])
    return input_data

n = len(training_data)
d = len(training_data[0]) - 1
training_data = convert_to_float(training_data)
testing_data = convert_to_float(testing_data)

def calculate_cost_function(w, input_data):
    loss = 0.5 * sum([(elem[-1] - np.inner(w, elem[0:d])) ** 2 for elem in input_data])
    return loss

def calculate_gradient(w, input_data):
    grad = []
    for j in range(d):
        grad.append(-sum([(row[-1] - np.inner(w, row[0:d])) * row[j] for row in input_data]))
    return grad

def batch_gradient(tol, rate, w, input_data):
    loss_vector = []
    while np.linalg.norm(calculate_gradient(w, input_data)) >= tol:
        loss_vector.append(calculate_cost_function(w, input_data))
        w = w - [rate * x for x in calculate_gradient(w, input_data)]
    return [w, loss_vector]

def calculate_stochastic_gradient(weight, sample_data, input_data):
    SG_vector = []
    for k in range(d):
        SG_vector.append(-(input_data[sample_data][-1] - np.inner(weight, input_data[sample_data][0:d])) * input_data[sample_data][k])
    return SG_vector

def find_SGD(tol, rate, weight, input_data, perm_seed):
    YesNo = 0
    loss_vector = []
    for num in perm_seed:
        if np.linalg.norm(calculate_stochastic_gradient(weight, perm_seed[num], input_data)) <= tol:
            YesNo = 1
            return [weight, loss_vector, YesNo]
        loss_vector.append(calculate_cost_function(weight, input_data))
        weight = weight - [rate * num for num in calculate_stochastic_gradient(weight, perm_seed[num], input_data)]
    return [weight, loss_vector, YesNo]

def permute_SGD(tol, rate, weight, input_data, reps):
    total_loss_vector = []
    for i in range(reps):
        perm_seed = np.random.permutation(n)
        [weight, loss_vec, YesNo] = find_SGD(tol, rate, weight, input_data, perm_seed)
        if YesNo == 1:
            return [weight, total_loss_vector]
        total_loss_vector = total_loss_vector + loss_vec
    return [weight, total_loss_vector]
##############################################################################################################
##############################################################################################################



# uncomment one of the following blocks to get the output for question 4 a-c


'''
##################################### Batch Gradient Method  (Question 4a) ##################################
[weight_vector, loss_vector] = batch_gradient(0.000001, 0.01, np.zeros(d), training_data)
print('the learned weight vector is',weight_vector)
print('cost function for the training dataset is:',calculate_cost_function(weight_vector, training_data))
print('cost function for the testing dataset is:',calculate_cost_function(weight_vector, testing_data))
plot.plot(loss_vector)
plot.ylabel('cost function value')
plot.xlabel('number of steps')
plot.title('Batch Gradient Method')
plot.show()
##############################################################################################################
'''


'''
##################################### Stochastic Gradient Method  (Question 4b) ##################################
[weight_vector, cost_vector] = permute_SGD(0.000001, 0.002, np.zeros(d), training_data, 10000)
print('The learned weight vector is :',weight_vector)
print('cost function for the training dataset is:',calculate_cost_function(weight_vector, training_data))
print('cost function for the testing dataset is:',calculate_cost_function(weight_vector, testing_data))
plot.plot(cost_vector)
plot.ylabel('cost function value')
plot.xlabel('number of steps')
plot.title('Stochastic Gradient Decent')
plot.show()
##############################################################################################################
'''

#'''
##################################### Analytical Solution (Question 4c) ##################################
feature_vector = [elem[0:d] for elem in training_data]
label_vector = [elem[-1] for elem in training_data]
matrix = np.array(feature_vector)
Y = np.array(label_vector)
X = matrix.transpose()
A = inv(np.matmul(X, X.transpose()))
B = np.matmul(A, X)
w_star =np.matmul(B, Y)
print(w_star)
##############################################################################################################
#'''

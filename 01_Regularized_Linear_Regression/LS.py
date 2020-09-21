import numpy as np
from numpy.linalg import linalg
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser() #initialize argument parser

#parse arguments from the command line
parser.add_argument('-A') #training data
parser.add_argument('-y') #target data
parser.add_argument('-beta', default=0.001) #regularization term
parser.add_argument('-x', default='output.txt') #output file path
parser.add_argument('-lr', default=0.01) #learning rate
parser.add_argument('-maxiters', default=1000) #max iterations allowed
parser.add_argument('-tol', default=1E-8) #stop if update smaller than tolerance

default = ['-Atrain_data2.txt','-ytrain_target2.txt']
args = parser.parse_args(default) #(default)

#store user specified arguments into variables
train_file = str(args.A)
target_file = str(args.y)
beta = float(args.beta)
output = str(args.x)
lr = float(args.lr)
max_iters = int(args.maxiters)
tol = float(args.tol)

#load feature matrix (training inputs) and target vector (regression targets)
feature = np.genfromtxt(train_file, delimiter=' ')
target = np.genfromtxt(target_file, delimiter=' ')

#set up variables for the SGD algorithm
weight = np.zeros((np.size(feature,axis=1),)) #initialize weight vector to all zeros
i = 0
'''
# of iterations allowed should be the minimum of (maximum iteration specified by the user, 
and the # of observations in our dataset - i.e. the # of rows in feature matrix)
'''
iter_limit = min(max_iters,len(feature))
history = [] #initialize a list called history to record the update history for visualization

while i < iter_limit: #keep SGD updates until reaching the maximum # of iterations allowed
    #compute the gradient of the loss with respect to w
    grad = (linalg.dot(np.transpose(weight),feature[i]) - target[i]) * feature[i] + beta * weight
    update = lr * grad #compute amount needs to be updated to the current weight vector
    weight = weight - update #SGD update
    history.append(update.mean()) #log the average update magnitude
    #If the absolute value of average update is smaller than our tolerance, stop the SGD process
    if np.absolute(update.mean()) < tol: break
    i += 1

#print("Last 10 updates:")
#print(history[-1:-10:-1])
np.savetxt(output, weight, delimiter=' ') #save final weight vector to output path

#visualizing the update process, see if converges
plt.plot(history)
plt.plot([0, len(history)],[0,0])
plt.show()
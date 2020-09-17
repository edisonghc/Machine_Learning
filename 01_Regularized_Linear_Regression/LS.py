import numpy as np
from numpy.linalg import linalg
#import pandas as pd
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-A')
parser.add_argument('-y')
parser.add_argument('-beta', default=0.001)
parser.add_argument('-x', default='output.txt')
parser.add_argument('-lr', default=0.01)
parser.add_argument('-maxiters', default=1000)
parser.add_argument('-tol', default=1E-8)

default = ['-Atrain_data2.txt','-ytrain_target2.txt']
args = parser.parse_args(default)

train_file = str(args.A)
target_file = str(args.y)
beta = float(args.beta)
output = str(args.x)
lr = float(args.lr) # learn rate
max_iters = int(args.maxiters) # max iterations
tol = float(args.tol) # tolerance

feature = np.genfromtxt(train_file, delimiter=' ')
target = np.genfromtxt(target_file, delimiter=' ')

weight = np.zeros((np.size(feature,axis=1),))
i = 0
iter_limit = min(max_iters,len(feature))
history = []

while i < iter_limit:
    # compute the gradient of the loss with respect to w
    grad = (linalg.dot(np.transpose(weight),feature[i]) - target[i]) * feature[i] + beta * weight
    update = lr * grad
    weight = weight - update # sgd update
    history.append(update.mean())
    if np.absolute(update.mean()) < tol: break
    i += 1

print("Last 10 updates:")
print(history[-1:-10:-1])
np.savetxt(output, weight, delimiter=' ')
plt.plot(history)
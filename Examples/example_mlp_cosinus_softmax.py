from AIScratch.NeuralNetwork import MLP
from AIScratch.NeuralNetwork import DenseLayer
from AIScratch.NeuralNetwork import MinMaxEncoder, Encoder
from AIScratch.NeuralNetwork import MSE, CrossEntropy, HingeLoss
from AIScratch.NeuralNetwork import Sigmoïde, Softmax, ReLU
from AIScratch.NeuralNetwork import SGDOptimizer, ADAMOptimizer
from random import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

"""
Example of usage of a perceptron of the library.
In this case, we are trying to find separate a set of points into
two groups. Points are labeled and the perceptron should figure out
the way to separate them.
"""
# points labeling
def labeling(xmin, xmax, ymin, ymax, number, func, error = 1):
    ret = {}
    for i in range(number):
        coo = ((xmax - xmin) * random() + xmin, (ymax - ymin) * random() + ymin)
        if error != 1 and random() > error:
            ret[coo] = [1,0] if coo[1] > func(coo[0]) else [0,1]
            continue
        ret[coo] = [0,1] if coo[1] > f(coo[0]) else [1,0]
    return ret

# training 
def training(network : MLP, training_set : dict[tuple[float, float], float], epoch_number : int):
    for epoch in range(epoch_number):
        print("Training :", epoch)
        for key in training_set.keys():
            inputs = np.array(key)
            network.backward(inputs, training_set[key])

# compute groups
def groups(set : dict[tuple[float, float], float], network : MLP = None, encoder : Encoder = None):
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for key in set.keys():
        _input = np.array(key) if encoder is None else encoder.encode(key)
        if (network == None and set[key] == [1,0]) or (network != None and network.forward(_input)[0]) > 0.9: # allow to get point either from label or from predictions
            x1.append(key[0])
            y1.append(key[1])
            continue
        x2.append(key[0])
        y2.append(key[1])
    return x1, y1, x2, y2

# compute network performance
def performance(set : dict[tuple[float, float], float], network : MLP, encoder : Encoder = None):
    ret = 0
    for key in set.keys():
        inputs = np.array(key) if encoder is None else encoder.encode(key)
        if network.forward(inputs)[0] < 0.1 and set[key] == [0,1]:
            ret += 1
        if network.forward(inputs)[0] > 0.9 and set[key] == [1,0]:
            ret += 1
    return ret / len(set.keys())

# plot results
def plot_results(xs_list,ys_list,labels,colors, plots, xlim = None, ylim = None, save_path = ""):
    plt.clf()
    for i in range(len(xs_list)):
        x = xs_list[i]
        y = ys_list[i]
        plot = plots[i]
        color = colors[i]
        label = labels[i]
        if plot:
            plt.plot(x, y, color, label=label)
        else:
            plt.scatter(x,y,c=color,label=label)  
    # regular parameters
    plt.grid()
    plt.legend()  
    if xlim != None:
        plt.xlim(xlim)
    if ylim != None:
        plt.ylim(ylim)
    if save_path != "":
        plt.savefig(save_path)
    else:
        plt.show()

# parameters
num_of_points = 1000
epoch_number = 50
# function def
f = lambda x : 0.25 * np.cos(x * 10) + 0.5 
x_sep = [x / 100 for x in range(0,101)]
y_sep = [f(x) for x in x_sep]

x_sep_test = [x / 100 for x in range(300,401)]
y_sep_test = [f(x) for x in x_sep_test]
# sets definitions
training_set = labeling(0,1,0,1, num_of_points, f)
test_set = labeling(3,4,-0.5,1.5, num_of_points, f)
# network definition
n_in = 2
#?-------------------#
sig = Sigmoïde()
soft = Softmax()
relu = ReLU()
#?-------------------#
ef = CrossEntropy()
#?-------------------#
test_encoder = MinMaxEncoder(np.array([4,1.5]),np.array([3,-0.5]))
#?-------------------#
# ADAM Optimizer
optimizer_factory = lambda n_p1, n_p : ADAMOptimizer(n_p1, n_p, 0.001, 1e-8, 0.9, 0.999)
# SGD Optimizer
# optimizer_factory = lambda n_p1, n_p : SGDOptimizer(n_p1, n_p, 0.001)
#?-------------------#
load = False
if load:
    layers = []
    mlp = MLP(n_in, layers, ef, optimizer_factory)
    mlp.load("Examples/mlp_cos_soft_network.txt")
else:
    layers = [DenseLayer(32, relu), DenseLayer(16, relu), DenseLayer(2, soft)]
    mlp = MLP(n_in, layers, ef, optimizer_factory, batch_size=1)
    training(mlp, training_set, epoch_number)
    
mlp.extract("Examples/mlp_cos_soft_network.txt")

# # success rate
print("Success rate on training : ", performance(training_set, mlp) * 100, "%")
print("Success rate on test : ", performance(test_set, mlp, test_encoder) * 100, "%")
x1,y1,x2,y2 = groups(training_set)
plot_results(
    [x1, x2, x_sep],
    [y1, y2, y_sep],
    ["Training group 1", "Training group 2", "Separation line"], 
    ["b", "r", "k"], 
    [False, False, True],
    (0,1), 
    (0,1), "MLP_training_set.png")
x1_pred,y1_pred,x2_pred,y2_pred = groups(training_set, mlp)
plot_results(
    [x1_pred, x2_pred, x_sep],
    [y1_pred, y2_pred, y_sep],
    ["Prediction group 1", "Prediction group 2", "Separation line"], 
    ["b", "r", "k"], 
    [False, False, True],
    (0,1), 
    (0,1), "MLP_training_pred.png")
x1_test,y1_test,x2_test,y2_test = groups(test_set, mlp, test_encoder)
plot_results(
    [x1_test, x2_test, x_sep_test],
    [y1_test, y2_test, y_sep_test],
    ["Test group 1", "Test group 2", "Separation line"], 
    ["b", "r", "k"], 
    [False, False, True],
    (3,4), 
    (-0.5,1.5), "MLP_training_test.png")

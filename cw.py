# import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import torch.nn.functional as F

def fitness(x1, x2):
    f = np.sin(3.5*x1 + 1)*np.cos(5.5*x2)
    # f = 2 + 4.1*(x1**2) - 2.1*(x1**4) + (1/3)*(x1**6) + (x1*x2) - 4*((x2-0.05)**2) + 4*(x2**4) 
    return (f)

def plot3D():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X = np.linspace(-1, 1, 100)
    Y = np.linspace(-1, 1, 100)
    # print(X)
    # Z = fitness(X, Y)
    # 
    # print(Z) 
    X, Y = np.meshgrid(X, Y)
    Z = fitness(X, Y)
    
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False, zorder=0)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('fitness')
    plt.title('3D Surface plot of the fitness function')
    plt.show()

def generate1100SamplesforX1andX2():
    X1 = np.random.uniform(-1, 1, 1100)
    X2 = np.random.uniform(-1, 1, 1100)
    Y = fitness(X1, X2)
    return X1, X2, Y

def splitDataInto1000TrainingAnd100TestingSamples(X1, X2, Y):
    X1_train = torch.from_numpy(X1[:1000])
    X2_train = torch.from_numpy(X2[:1000])
    Y_train = torch.from_numpy(Y[:1000])
    # print(X1_train)
    X1_test = torch.from_numpy(X1[1000:])
    X2_test = torch.from_numpy(X2[1000:])
    Y_test = torch.from_numpy(Y[1000:])
    # print(type(Y_test), type(Y_train), type(X1_test), type(X1_train))
    training = (X1_train, X2_train, Y_train)
    testing = (X1_test, X2_test, Y_test)
    
    return training, testing

def visualizeTrainingandTesting(training, testing):
    X1_train, X2_train, Y_train = training
    X1_test, X2_test, Y_test = testing
    fig = plt.figure(figsize=(26, 6))
    ax = fig.add_subplot(1,2,1, projection='3d')
    ax.scatter(X1_train, X2_train, Y_train, c='blue', marker='o')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('fitness')
    plt.title('3D plot of training dataset')
    # ax.scatter(X1_test, X2_test, Y_test, c='black', marker='o')
    ax = fig.add_subplot(1,2,2, projection='3d')
    ax.scatter3D(X1_test, X2_test, Y_test, c='red', marker='o')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('fitness')
    plt.title('3D plot of testing dataset')
    plt.show()

class Net(torch.nn.Module):
    # initialise two hidden layers and one output layer
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.hidden2 = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)  # output layer

    # connect up the layers: the input passes through the hidden, then the sigmoid, then the output layer
    def forward(self, x):
        x = F.sigmoid(self.hidden(x))
        x = F.sigmoid(self.hidden2(x))  # activation function for hidden layer
        x = self.out(x)
        return x

# Plots 3D surface plot of function
plot3D()

# Generates dataset
dataset = generate1100SamplesforX1andX2()

# Splits dataset into training and testing dataset
training, testing = splitDataInto1000TrainingAnd100TestingSamples(*dataset)

# Visualize training and testing dataset
visualizeTrainingandTesting(training, testing)

# Create model with two hidden layers and 6 neurons in each
net = Net(n_feature=2, n_hidden=6, n_output=1)
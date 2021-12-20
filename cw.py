import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from torch.functional import Tensor
import torch.nn.functional as F
from sympy.combinatorics.graycode import GrayCode
from sympy.combinatorics.graycode import gray_to_bin, bin_to_gray
from deap import creator, base, tools, algorithms
import copy

class Net(torch.nn.Module):
    # initialise two hidden layers and one output layer
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)  # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)  # output layer

    # connect up the layers: the input passes through the hidden, then the sigmoid, then the output layer
    def forward(self, x):
        x = F.sigmoid(self.hidden(x)) # activation function for hidden layer
        x = F.sigmoid(self.hidden2(x))  # activation function for hidden layer 2
        x = self.out(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # cuda is the GPU

loss_func = torch.nn.MSELoss()
totalBits = 67*30 # [(input size + 1) * numOfHiddenNeurons + (numOfHiddenNeurons + 1) * output] * numOfBitsPerWeight 
popSize = 50
dimension = 67 # number of dimensions
numOfBits = 30 # number of bits per weight
numOfGenerations = 30 # number of generations to run
nElitists = 1
crossPoints = 2 #variable not used. instead tools.cxTwoPoint
crossProb   = 0.6
flipProb    = 1. / (dimension * numOfBits) #bit mutate prob
mutateprob  = .1 #mutation prob
maxnum      = 2**numOfBits-1

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# intitialize neural network to GPU
model = Net(n_feature=2, n_hidden=6, n_output=1).to(device)
torch.save(model, 'model.pt') # save model to folder
toolbox = base.Toolbox()

# initial y function 
def fitness(x1, x2):
    f = np.sin(3.5*x1 + 1)*np.cos(5.5*x2)
    # f = 2 + 4.1*(x1**2) - 2.1*(x1**4) + (1/3)*(x1**6) + (x1*x2) - 4*((x2-0.05)**2) + 4*(x2**4) 
    return (f)

# 3D surface plot of fitness function
def plot3D():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x1 = np.linspace(-1, 1, 100)
    x2 = np.linspace(-1, 1, 100)
    x1, x2 = np.meshgrid(x1, x2)
    Z = fitness(x1, x2)
    
    ax.plot_surface(x1, x2, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False, zorder=0)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('fitness')
    plt.title('3D Surface plot of the fitness function')
    plt.show()

# generates 1100 samples of x1 and x2 values
# returns x1, x2, and y (output from the fitness function for every x1 and x2)
def generate1100SamplesforX1andX2():
    X1 = np.random.uniform(-1, 1, 1100)
    X2 = np.random.uniform(-1, 1, 1100)
    Y = fitness(X1, X2)
    return X1, X2, Y

# splits x1 and x2 into training and testing datasets
# returns tuple of training and testing data
def splitDataInto1000TrainingAnd100TestingSamples(X1, X2, Y):
    
    # Training dataset
    # transforms x1 and x2 into 2d array of tensors
    X_train = torch.from_numpy(np.vstack((X1[:1000], X2[:1000])).T).to(device)
    # transforms y (output) into a tensor
    Y_train = torch.from_numpy(Y[:1000]).to(device)

    # Testing dataset
    # transforms x1 and x2 into a 2d array of tensors
    X_test = torch.from_numpy(np.vstack((X1[1000:], X2[1000:])).T).to(device)
    # transforms y (output) into a tensor
    Y_test = torch.from_numpy(Y[1000:]).to(device)
    training = (X_train, Y_train)
    testing = (X_test, Y_test)
    
    return training, testing

# creates 3D scatter plot of the training and testing dataset
def visualizeTrainingandTesting(training, testing):
    X_train, Y_train = training
    X_test, Y_test = testing
    fig = plt.figure(figsize=(26, 6))
    ax = fig.add_subplot(1,2,1, projection='3d')
    for i in range(len(X_train)):
        ax.scatter3D(X_train[i][0], X_train[i][1], Y_train[i], c='blue', marker='o')
        # plt.pause(0.000001)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('fitness')
    ax.view_init(80, 30)
    plt.title('3D plot of training dataset')
    # ax.scatter(X1_test, X2_test, Y_test, c='black', marker='o')
    ax = fig.add_subplot(1,2,2, projection='3d')
    for i in range(len(X_test)):
        ax.scatter3D(X_test[i][0], X_test[i][1], Y_test[i], c='red', marker='o')
        # plt.pause(0.000001)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('fitness')
    plt.title('3D plot of testing dataset')
    plt.show()

# extract weights out of neural network
# returns array of weights
def extractWeightsOutOfNetwork(nn):
    outweights = []
    for param in nn.parameters():
        data = Tensor.cpu(param.data) # convert cpu to tensor to perform numpy operations
        flattened = (np.array(data).flatten()).tolist()
        outweights += flattened
    return outweights

# inputs weights into neural network
# input: array of weights, network being modified
def inputWeightsIntoNetwork(arr, nn):
    weights = np.asarray(arr)
    nn.hidden.weight = torch.nn.Parameter(torch.from_numpy(weights[:12].reshape(6, 2)).to(device))
    nn.hidden.bias =  torch.nn.Parameter(torch.from_numpy(weights[12:18].reshape(1, 6)).to(device))
    nn.hidden2.weight = torch.nn.Parameter(torch.from_numpy(weights[18:54].reshape(6, 6)).to(device))
    nn.hidden2.bias =  torch.nn.Parameter(torch.from_numpy(weights[54:60].reshape(1, 6)).to(device))
    nn.out.weight = torch.nn.Parameter(torch.from_numpy(weights[60:66].reshape(1, 6)).to(device))
    nn.out.bias = torch.nn.Parameter(torch.from_numpy(weights[66:67].reshape(1, 1)).to(device))
    return nn

# creates 3D surface plot of the fitness function
def neuralNetwork3DSurfacePlot():
    X = np.linspace(-1, 1, 100)
    Y = np.linspace(-1, 1, 100)
    combined = torch.from_numpy(np.vstack([X, Y]).T).to(device)

    X, Y = np.meshgrid(X, Y)
    Z = model(combined)
    Z = Tensor.cpu(Z).detach()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False, zorder=0)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    plt.title('3D surface plot of fitness function implemented by the neural network')
    plt.show()

def plot(maxArr):
    print("plotting................................................................")
    # maxArr = maxArr.detach().numpy()
    gen = []
    for i in range(numOfGenerations):
        gen.append(i)

    plt.plot(gen, maxArr, label="Best Individual")
    # plt.plot(gen, avgArr, label="Average Individual")
    plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness of best individual from the testing dataset across the generations")
    plt.show()

# Convert chromosome to real number
# input: list binary 1,0 of length numOfBits representing number using gray coding
# output: real value
def chrom2real(c):
    indasstring=''.join(map(str, c))
    degray=gray_to_bin(indasstring)
    numasint=int(degray, 2) # convert to int from base 2 list
    numinrange=-20+40*numasint/maxnum
    return numinrange

# Convert array of weights to chromosome
# input: array of weights
# output: list binary 1,0 of length numOfBits representing number using gray coding
def real2Chrom(weights):
    output = [] # create output array to hold individual
    for i in range(len(weights)): # clamp weights between -20 and 20
        if weights[i] < -20:
            weights[i] = -20
        elif weights[i] > 20:
            weights[i] = 20
        numasint = (weights[i] + 20)*maxnum/40 # convert weight to integer
        binary = bin(int(numasint))[2:].zfill(30) # convert to binary
        gray = bin_to_gray(binary) # convert to gray-coded digit
        output.append(gray)
    output = list(''.join(output))
    for i in range(len(output)):
        output[i] = int(output[i])
    return output

# Return loss value of the neural network on the current individual
# input: list binary 1,0 of length numOfBits representing number using gray coding
# output: loss value of individual
def getWeightFitness(individual):
    individual = np.array(individual) # convert
    reshaped = individual.reshape(67, 30) # reshape to array of weights
    weights = []
    for ind in reshaped:
        ind = chrom2real(ind)
        weights.append(ind)
    weights = np.asarray(weights) # create array of weights as real numbers
    inputWeightsIntoNetwork(weights, model) # update model with new weights
    out = model(training[0]) # input training data and predict network output based on data
    loss = loss_func(out, training[1]) # # compare output with labeled data
    return loss.item(),

# Implements lamarckian learning local search on each individual
# input: list binary 1,0 of length numOfBits representing number using gray coding
# output: updated list binary 1,0 of length numOfBits representing number using gray coding
def lamarckianOptimize(individual):
    individual = np.array(individual) # convert individual to numpy array
    reshaped = individual.reshape(67, 30) # reshape individual to 67x30
    weights = []
    for ind in reshaped:
        ind = chrom2real(ind) # convert weights to real number
        weights.append(ind)
    weights = np.asarray(weights)
    # print("original weights: ", extractWeightsOutOfNetwork(model))
    inputWeightsIntoNetwork(weights, model) # input weights into network
    y = model(training[0])
    originalLoss = loss_func(y, training[1]) # calculate loss
    i = 0
    loss = 0
    updatedWeights = extractWeightsOutOfNetwork(model) # extract weights from
    optimizer = torch.optim.Rprop(model.parameters(), lr=0.005) # intialize RProp optimizer
    while i < 30: # run 30 iterations of optimization to find better weights
        out = model(training[0]) # input training data and predict network output based on data
        loss = loss_func(out, training[1]) # compare output with labeled data
        optimizer.zero_grad() # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        if originalLoss > loss: # update weights if loss result is better
            updatedWeights = extractWeightsOutOfNetwork(model)
            originalLoss = loss
        i += 1
    newInd = real2Chrom(updatedWeights) # convert weights to gray coded individual
    return newInd

# Implements baldwinian learning local search on each individual
# input: list binary 1,0 of length numOfBits representing number using gray coding
# output: tuple loss value of optimized individual
def baldwinianLearning(individual):
    individual = np.array(individual) # convert individual to numpy array
    reshaped = individual.reshape(67, 30) # reshape individual to 67x30
    weights = []
    for ind in reshaped:
        ind = chrom2real(ind) # convert weights to real number
        weights.append(ind)
    weights = np.asarray(weights)
    inputWeightsIntoNetwork(weights, model) # input weights into network
    i = 0
    loss = 0
    y = model(training[0])
    originalLoss = loss_func(y, training[1]) # calculate loss
    optimizer = torch.optim.Rprop(model.parameters(), lr=0.005) # intialize RProp optimizer
    while i < 30:   
        optimizer.zero_grad() # clear gradients for next train
        out = model(training[0]) # input training data and predict network output based on data
        loss = loss_func(out, training[1]) # compare output with labeled data
        loss.backward() # backpropagation, compute gradients
        optimizer.step() # apply gradients
        i += 1
        if originalLoss > loss: # update original loss if loss result is better
            originalLoss = loss
    return originalLoss.item(),

def plotLearning(lamarckian, baldwinian):
    print("plotting................................................................")
    # maxArr = maxArr.detach().numpy()
    gen = []
    for i in range(numOfGenerations):
        gen.append(i)

    plt.plot(gen, lamarckian, label="Lamarckian Learning", color="blue")
    plt.plot(gen, baldwinian, label="Baldwinian Learning", color="green")
    plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness of best individual from across the generations")
    plt.show()

# Generates dataset
print("================================================Dataset================================================")
dataset = generate1100SamplesforX1andX2()

# Splits dataset into training and testing dataset
print("================================================Splitting Dataset================================================")
training, testing = splitDataInto1000TrainingAnd100TestingSamples(*dataset)
# print(type(training[0]))
# Visualize training and testing dataset
# visualizeTrainingandTesting(training, testing)

print("================================================Creating Network =================================================")
# Create model with two hidden layers and 6 neurons in each
net = Net(n_feature=2, n_hidden=6, n_output=1).to(device)

# Extracted weights from network
extractedWeights = extractWeightsOutOfNetwork(net)

# Test to see if new weights can be inputted
print("================================================Weights================================================")
print(extractedWeights)
print("================================================Weights from first layer================================================")
print(net.hidden.weight)
print(extractedWeights[:12])
print("================================================Changing 3 Weights==============================================")

# Change weights
extractedWeights[1] = 0.5
extractedWeights[2] = 0.4
extractedWeights[3] = 0.3
net = inputWeightsIntoNetwork(extractedWeights, net)

newWeights = extractWeightsOutOfNetwork(net)
print(newWeights)

print("================================================Running GA================================")

# Attribute generator 
#                      define 'attr_bool' to be an attribute ('gene')
#                      which corresponds to integers sampled uniformly
#                      from the range [0,1] (i.e. 0 or 1 with equal
#                      probability)
toolbox.register("attr_bool", random.randint, 0, 1)

# Structure initializers
#                         define 'individual' to be an individual
#                         consisting of numOfBits*dimension 'attr_bool' elements ('genes')
toolbox.register("individual", tools.initRepeat, creator.Individual, 
    toolbox.attr_bool, totalBits)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#----------
# Operator registration
#----------
# register the goal / fitness function
toolbox.register("evaluate", getWeightFitness)

# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutate", tools.mutFlipBit, indpb=flipProb)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selBest, fit_attr='fitness')

arr = []
bestInitial = 0

# Run GA optimizer for numOfGenerations generations
# inputs: x (population), boolean to decide local search method
def main(x, boolean):
    fitArr = []
    # print(pop)
    #random.seed(64)

    # create an initial population of individuals (where
    # each individual is a list of integers)
    pop = copy.deepcopy(x)
#     for individ in pop:
#         sep=separatevariables(individ)
#         print(sep[0],sep[1])

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    #print(fitnesses)
    for ind, fit in zip(pop, fitnesses):
        #print(ind, fit)
        ind.fitness.values = fit
    
    print("  Evaluated %i individuals" % len(pop))
    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]
    bestInitial = tools.selBest(pop, 1)[0].fitness.values[0]
    # Variable keeping track of the number of generations
    g = 0
    # print(fitnesses)
    # Begin the evolution
    while g < numOfGenerations:
        # A new generation
        
        # Run lamarckian learning on each generation
        if boolean is True:
            for ind in pop:
                ind = lamarckianOptimize(ind)
            fitnesses = list(map(toolbox.evaluate, pop))  
        
        # Run baldwinian learning on each generation
        else:
            fitnesses = list(map(baldwinianLearning, pop))
            
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        best_ind = tools.selBest(pop, 1)[0]
        fitnessBest = best_ind.fitness.values[0]
        arr.append(fitnessBest)    
        
        g = g + 1
        print("-- Generation %i --" % g)
        # Select the next generation individuals
        offspring = tools.selBest(pop, nElitists) + toolbox.select(pop,len(pop)-nElitists)
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        
#         for individ in offspring:
#             print(individ)

    
        # Apply crossover and mutation on the offspring
        # make pairs of offspring for crossing over
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < crossProb:
                #print('before crossover ',child1, child2)
                toolbox.mate(child1, child2)
                #print('after crossover ',child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability mutateprob
            if random.random() < mutateprob:
                toolbox.mutate(mutant)
                del mutant.fitness.values
    
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        #print("  Evaluated %i individuals" % len(invalid_ind))
        
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        
    if boolean is True:
        np.save('lamarckian', arr)
    else:
        np.save('baldwinian', arr)

if __name__ == "__main__":
    x = toolbox.population(n=popSize)
    x2 = copy.deepcopy(x)
    main(x, True)
    # neuralNetwork3DSurfacePlot()
    arr = []
    model = torch.load('model.pt').to(device)
    main(x2, False)
    # neuralNetwork3DSurfacePlot()
    print("================================================Extracting Weights out of Network================================================")
    extracted = extractWeightsOutOfNetwork(model)
    print(extracted)
    print("================================================Converting weights to individual==============================================")
    real = real2Chrom(extracted)
    print(real)
    l = np.load('lamarckian.npy')
    b = np.load('baldwinian.npy')
    print(l)
    print(b)
    plotLearning(l, b)
    plot(b)
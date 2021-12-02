import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import torch.nn.functional as F
from sympy.combinatorics.graycode import GrayCode
from sympy.combinatorics.graycode import gray_to_bin, bin_to_gray
from deap import creator, base, tools, algorithms

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

loss_func = torch.nn.MSELoss()
totalBits = 67*30 # [(input size + 1) * numOfHiddenNeurons + (numOfHiddenNeurons + 1) * output] * numOfBitsPerWeight 
popSize = 50
dimension = 67
numOfBits = 30
numOfGenerations = 30
nElitists = 1
crossPoints = 2 #variable not used. instead tools.cxTwoPoint
crossProb   = 0.5
flipProb    = 1. / (dimension * numOfBits) #bit mutate prob
mutateprob  = .1 #mutation prob
maxnum      = 2**numOfBits-1

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

model = Net(n_feature=2, n_hidden=6, n_output=1)

toolbox = base.Toolbox()

# initial y function 
def fitness(x1, x2):
    f = np.sin(3.5*x1 + 1)*np.cos(5.5*x2)
    # f = 2 + 4.1*(x1**2) - 2.1*(x1**4) + (1/3)*(x1**6) + (x1*x2) - 4*((x2-0.05)**2) + 4*(x2**4) 
    return (f)

# plots 3D space
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

def generate1100SamplesforX1andX2():
    X1 = np.random.uniform(-1, 1, 1100)
    X2 = np.random.uniform(-1, 1, 1100)
    Y = fitness(X1, X2)
    return X1, X2, Y

def splitDataInto1000TrainingAnd100TestingSamples(X1, X2, Y):
    X_train = torch.from_numpy(np.vstack((X1[:1000], X2[:1000])).T)
    Y_train = torch.from_numpy(Y[:1000])
    # print(X1_train)
    X_test = torch.from_numpy(np.vstack((X1[1000:], X2[1000:])).T)
    Y_test = torch.from_numpy(Y[1000:])
    # print(type(Y_test), type(Y_train), type(X1_test), type(X1_train))
    training = (X_train, Y_train)
    testing = (X_test, Y_test)
    
    return training, testing

def visualizeTrainingandTesting(training, testing):
    X_train, Y_train = training
    X_test, Y_test = testing
    fig = plt.figure(figsize=(26, 6))
    ax = fig.add_subplot(1,2,1, projection='3d')
    for i in range(len(X_train)):
        ax.scatter3D(X_train[i][0], X_train[i][1], Y_train[i], c='blue', marker='o')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('fitness')
    ax.view_init(80, 30)
    plt.title('3D plot of training dataset')
    # ax.scatter(X1_test, X2_test, Y_test, c='black', marker='o')
    ax = fig.add_subplot(1,2,2, projection='3d')
    for i in range(len(X_test)):
        ax.scatter3D(X_test[i][0], X_test[i][1], Y_test[i], c='red', marker='o')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('fitness')
    plt.title('3D plot of testing dataset')
    plt.show()

def extractWeightsOutOfNetwork(nn):
    outweights = []
    for param in nn.parameters():
        flattened = (np.array(param.data).flatten()).tolist()
        outweights += flattened
    return outweights

def inputWeightsIntoNetwork(arr, nn):
    weights = np.asarray(arr)
    nn.hidden.weight = torch.nn.Parameter(torch.from_numpy(weights[:12].reshape(6, 2)))
    nn.hidden.bias =  torch.nn.Parameter(torch.from_numpy(weights[12:18].reshape(1, 6)))
    nn.hidden2.weight = torch.nn.Parameter(torch.from_numpy(weights[18:54].reshape(6, 6)))
    nn.hidden2.bias =  torch.nn.Parameter(torch.from_numpy(weights[54:60].reshape(1, 6)))
    nn.out.weight = torch.nn.Parameter(torch.from_numpy(weights[60:66].reshape(1, 6)))
    nn.out.bias = torch.nn.Parameter(torch.from_numpy(weights[66:67].reshape(1, 1)))
    return net

def neuralNetwork3DSurfacePlot():
    X = np.linspace(-1, 1, 100)
    Y = np.linspace(-1, 1, 100)
    combined = torch.from_numpy(np.vstack([X, Y]).T)

    X, Y = np.meshgrid(X, Y)
    Z = model(combined)
    Z = Z.detach()
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
    plt.xlim(left=0)
    plt.ylim(bottom=0)
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

def real2Chrom(weights):
    output = []
    for i in range(len(weights)):
        if weights[i] < -20:
            weights[i] = -20
        elif weights[i] > 20:
            weights[i] = 20
        numasint = (weights[i] + 20)*maxnum/40
        binary = bin(int(numasint))[2:].zfill(30)
        gray = bin_to_gray(binary)
        output.append(gray)
    output = list(''.join(output))
    return output

def getWeightFitness(individual):
    individual = np.asarray(individual)
    reshaped = individual.reshape(67, 30)
    weights = []
    for ind in reshaped:
        ind = chrom2real(ind)
        weights.append(ind)
    weights = np.asarray(weights)
    inputWeightsIntoNetwork(weights, model)
    out = model(testing[0])  # input x and predict based on x
    loss = loss_func(out, testing[1])
    return 1/(loss.item() + 0.01),

# plot3D()

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
net = Net(n_feature=2, n_hidden=6, n_output=1)

# Extracted weights from network
extractedWeights = extractWeightsOutOfNetwork(net)

# Test to see if new weights can be inputed
print("================================================Original Weights================================================")
print(extractedWeights)
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

def main():
    #random.seed(64)

    # create an initial population of individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=popSize)
    
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

    # Variable keeping track of the number of generations
    g = 0
    
    # Begin the evolution
    while g < numOfGenerations:
        # A new generation
        g = g + 1
#         for individ in pop:
#             print(individ)
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
        best_ind = tools.selBest(pop, 1)[0]
        fitnessBest = 1 / best_ind.fitness.values[0]
        arr.append(fitnessBest)
        #print("  Evaluated %i individuals" % len(invalid_ind))
        
        # The population is entirely replaced by the offspring
        pop[:] = offspring
    plot(arr)
# train model on training dataset
# print("================================================Training Model================================================")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

if __name__ == "__main__":
    main()
    print(model.hidden.weight)
    neuralNetwork3DSurfacePlot()
    extracted = extractWeightsOutOfNetwork(model)
    print(extracted)
    real = real2Chrom(extracted)
    print(real)
import torch
import torch.nn.functional as F
from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np

# load the training data
data = genfromtxt('cancer_TR.dat', delimiter=' ')
#first 9 columns are input data (aka feature)
x = data[:, 0:9]
#last 2 columns are the labels (in fact one column would have been enough since it is 1 or 0)
y = data[:, 9:11]
print(x)
print(y)
#turn the data into a torch tensor
x = torch.as_tensor(x, dtype=torch.float32)
y = torch.as_tensor(y, dtype=torch.float32)


print(x.size())
print(y)
print(y.size())


# torch.manual_seed(1)    # reproducible experiments
                          # by fixing the seed you will remove randomness
######################
# set up the network #
######################

class Net(torch.nn.Module):
    # initialise one hidden layer and one output layer
    def __init__(self, n_feature, n_hidden, n_output):
        # call pytorch superclass code to initialise nn
        super(Net, self).__init__()
        # hidden layer
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  
        self.out = torch.nn.Linear(n_hidden, n_output)  # output layer

    # connect up the layers: the input passes through the hidden, then the sigmoid, then the output layer
    # x is the data
    def forward(self, x):
        # activation function for hidden layer is relu
        # data passes through hidden and relu
        x = F.relu(self.hidden(x))  
        # the result needs to pass through output layer next
        x = self.out(x)
        return x

net = Net(n_feature=9, n_hidden=2, n_output=2)  # define the network

#optional printouts about network
print("printing net")
print(net)  # net architecture


######################
# Train the network  #
######################

# What optimiser to use, e.g. SGD or Rprop
optimizer = torch.optim.SGD(net.parameters(), lr=0.02) #lr = learning rate
loss_func = torch.nn.MSELoss() #MSE = mean squared error
loss_values = []

#do 600 steps of gradient descent
for t in range(600):
    out = net(x)  # input x and predict network output based on x
    loss = loss_func(out, y) # compare output with grount truth label y
    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients
    loss_values.append(loss.item()) # keep track of loss values for later plot
# it is good practice to save the network when trained
torch.save(net.state_dict(), 'net_params.pkl')  # save
plt.plot(np.array(loss_values), 'r')
plt.show()

######################
# Test the network   #
######################

#create a second copy of the netwrk for testing (not really necessary)
net2 = Net(n_feature=9, n_hidden=2, n_output=2)
net2.load_state_dict(torch.load('net_params.pkl'))
# load the test data
test = genfromtxt('cancer_tt.dat', delimiter=' ')
x2 = test[:, 0:9]
y2 = test[:, 9:11]
x2 = torch.as_tensor(x2, dtype=torch.float32)
y2 = torch.as_tensor(y2, dtype=torch.float32)

#predicted_indices will give us the winning class
_, predicted_indices = torch.max(net2(x2), 1) # Returns a namedtuple (values, indices) 
                                # where values is the maximum value of each row of the input tensor 
                                # indices is the index location of each maximum value found (argmax).
#target_indices will give us the ground truth label
_, target_indices = torch.max(y2, 1)
total = y2.size(0)
#sum up how many we got right, divide by total
accuracy = (predicted_indices == target_indices ).sum().item() / total
print("accuracy:", accuracy)
print("\nMake a prediction for a sample input:")
print("input:",x2[10])
predicted = net2(x2[10])
print("predicted:",predicted)
print("actual:",y2[10])
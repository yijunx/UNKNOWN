import numpy as np
import matplotlib.pyplot as plt
# generate data

np.random.seed(42)

x = np.random.rand(100, 1)  # x is 100 * 1, 2D matrix
y = 1 + 2 * x + 0.1 * np.random.randn(100, 1)

# shuffle the index
idx = np.arange(100)
np.random.shuffle(idx)

# use first 80 random index for train
train_idx = idx[:80]
# use the remaining ones for validation
val_idx = idx[80:]

# generate the train sets and validation sets
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

# plt.scatter(x_train, y_train)
# plt.show()

# step 1 ===== computing the loss
# for this regression problem, loss is given by the MSE
# MSE = 1 / n * sigma_i=1_to_n{(y_i - y_i_predict)**2}

# random initialization of parameters/weights (a and b)
# initialization of hyper parameters (the learning rate eta and number of epochs)



a = np.random.randn(1)
b = np.random.randn(1)

print(f'a is {a}, and b is {b}')

# learning rate and epochs
lr = 1e-1  # 0.1
n_epochs = 1000

print(f'learning rate is {lr}, number of epochs is {n_epochs}')

for epoch in range(n_epochs):  # the iterations to train
    # forward
    yhat = a + b * x_train

    error = (y_train - yhat)

    # mse loss functions
    loss = (error ** 2).mean()

    # the gradient
    a_grad = -2 * error.mean()
    b_grad = -2 * (x_train * error).mean()

    # move on the gradient by learning rate to update a and b
    a = a - lr * a_grad
    b = b - lr * b_grad

print(f'after train, a is {a}, and b is {b}')


# Sanity Check: do we get the same results as our gradient descent?
# from sklearn.linear_model import LinearRegression
# linr = LinearRegression()
# linr.fit(x_train, y_train)
# print(linr.intercept_, linr.coef_[0])



import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torch.nn.functional as F


class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        # affects self[index]
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# create dataset
x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()
dataset = TensorDataset(x_tensor, y_tensor)


train_dataset, val_dataset = random_split(dataset, [80, 20])

train_loader = DataLoader(dataset=train_dataset, batch_size=16)
val_loader = DataLoader(dataset=val_dataset, batch_size=20)

# print(next(iter(train_loader)))

# print(type(x_train), type(x_train_tensor), x_train_tensor.type(), type(x_train_tensor.numpy()))


lr = 0.1
n_epochs = 1000
torch.manual_seed(42)

print(a, b)

# now lets make a model

class my_model_for_linear_req(nn.Module):
    def __init__(self):
        super().__init__()  # because this guys is subclass of nn.Module
        # self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        # self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        # Instead of our custom parameters, we use a Linear layer with single input and single output
        self.linear = nn.Linear(1, 1)  # 1 feature in, 1 feature out

    def forward(self, x):
        return self.linear(x)

class my_model_cnn_trend_1d_convo(nn.Module):
    def __init__(self):
        super().__init__()  # because this guys is subclass of nn.Module
        # self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        # self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        # Instead of our custom parameters, we use a Linear layer with single input and single output
        # self.linear = nn.Linear(1, 1)  # 1 feature in, 1 feature out

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(3, 1))
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=(3, 1))

        # fully connected layers
        self.fc1 = nn.Linear(10 * 8 * 5, 80)
        self.fc2 = nn.Linear(80, 10)
        self.fc3 = nn.Linear(10, 2)



    def forward(self, x):
        return self.linear(x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# let's encapsulate the model and loss fn and optimizer
# we can do a make test step here also

def make_train_step(some_model, some_loss_fn, some_optimizer):

    def train_step(x, y):
        # set the model into train mode
        some_model.train()
        # forward
        yhat = some_model.forward(x)
        # get the loss
        loss = some_loss_fn(y, yhat)
        # get the grade
        loss.backward()
        # update it
        some_optimizer.step()
        some_optimizer.zero_grad()
        # return the loss
        return loss.item()

    return train_step

# now create the model
a_model = my_model_for_linear_req().to(device)
# define the optimizer
an_optimizer = optim.SGD(a_model.parameters(), lr=lr)
# define the loss function
a_loss_fn = nn.MSELoss(reduction='mean')
# create the function based on the parameters
train_step = make_train_step(a_model, a_loss_fn, an_optimizer)

losses = []
val_losses = []

for epoch in range(n_epochs):

    # # Its only purpose is to set the model to training mode
    # a_model.train()
    #
    # # forward (replaced by model forward)
    # # yhat = a + b * x_train_tensor
    # yhat = a_model.forward(x_train_tensor)
    #
    # # error and loss (replaced by loss fn)
    # # error = y_train_tensor - yhat
    # # loss = (error ** 2).mean()
    # loss = loss_fn(y_train_tensor, yhat)  # first input(given), then target(predicted)
    #
    # # now tell pytorch to work its way back to calculate the grad!!
    # # do the differentiation and move along the gradient
    # # replaced by backward() and optimizer.step()
    # # lets check the computed gradients
    # # print(f'the {epoch + 1} iteration: ')
    # # print(f'a_grad is {a.grad}, b_grad is {b.grad}')
    # # with torch.no_grad():
    # #     a -= lr * a.grad
    # #     b -= lr * b.grad
    # # ask the optimizer to move a step for a and b along the gradient
    # loss.backward()
    # optimizer.step()
    #
    # # PyTorch is "clingy" to its computed gradients, we need to tell it to let it go...
    # # print('after moving with lr')
    # # print(a)
    # # print(b)
    # # a.grad.zero_()
    # # b.grad.zero_()
    # optimizer.zero_grad()

    for x_batch, y_batch in train_loader:
        loss = train_step(x_batch, y_batch)
        # print(x_batch.size())
        losses.append(loss)

    with torch.no_grad():
        for x_val, y_val in val_loader:
            a_model.eval()

            yhat = a_model(x_val)
            val_loss = a_loss_fn(y_val, yhat)
            val_losses.append(val_loss)

print(a_model.state_dict())
# print(f'len loss is {len(losses)}')

# torch.manual_seed(42)
# a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
# b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
#
# yhat = a + b * x_train_tensor
# error = y_train_tensor - yhat
# loss = (error ** 2).mean()
#
# make_dot(yhat)







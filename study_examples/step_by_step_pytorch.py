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

x_train_tensor = torch.from_numpy(x_train).float()  # .to(device)
y_train_tensor = torch.from_numpy(y_train).float()  # .to(device)

train_data = CustomDataset(x_train_tensor, y_train_tensor)
print(train_data[0])

train_data = TensorDataset(x_train_tensor, y_train_tensor)
print(train_data[0])

train_loader = DataLoader(dataset=train_data, batch_size=20, shuffle=True)

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


# let's encapsulate the model and loss fn and optimizer


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







import numpy as np
import model
import pandas as pd

# reload Model each time this thing is run in order to test
import importlib
importlib.reload(model)


tf_gen_0 = pd.Series(index=['model_name'],
                     data=['tf_gen_0'])

m = model.Model('economics_False_by_week.csv',
          'SPY_end_at_2020-2-23_for_100_weeks.csv',
          tf_gen_0,
          'close_open',
          'somefilenamehere')

# now it is the time for the
m.form_X_y(weeks_to_predict=30, scaled=False, div_100=True, flatten=False)


# now it is the time for pyTorch!

import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# create dataset
x_tensor = torch.from_numpy(m.X).float()
y_tensor = torch.from_numpy(m.y).float()
dataset = TensorDataset(x_tensor, y_tensor)

train_dataset, val_dataset = random_split(dataset, [50, 20])

train_loader = DataLoader(dataset=train_dataset)  #, batch_size=10)
val_loader = DataLoader(dataset=val_dataset)  #, batch_size=10)

# lr = 0.01
n_epochs = 1000

# the model
class my_model_cnn_trend_1d_convo(nn.Module):
    def __init__(self):
        super().__init__()  # because this guys is subclass of nn.Module
        # self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        # self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        # Instead of our custom parameters, we use a Linear layer with single input and single output
        # self.linear = nn.Linear(1, 1)  # 1 feature in, 1 feature out

        self.conv1 = nn.Conv1d(in_channels=5, out_channels=5, kernel_size=6)

        # each channel is a kw, in channel=number of kw
        # we do the process of convo...

        # weeks to predict = w,
        self.conv2 = nn.Conv1d(in_channels=5, out_channels=5, kernel_size=6)

        # now the result is 5 * (30 - (5-1) - (5-1))

        # fully connected layers
        self.fc1 = nn.Linear(5 * (30 - (6-1) - (6-1)), 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # flatten it
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



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
a_model = my_model_cnn_trend_1d_convo().to(device)
# define the optimizer

a_loss_fn = nn.MSELoss(reduction='mean')
an_optimizer = optim.SGD(a_model.parameters(), lr=0.001, momentum=0.9)

# an_optimizer = optim.SGD(a_model.parameters(), lr=lr)
# define the loss function
# a_loss_fn = nn.MSELoss(reduction='mean')
# create the function based on the parameters
train_step = make_train_step(a_model, a_loss_fn, an_optimizer)

losses = []
val_losses = []

for epoch in range(n_epochs):
    print(f'Round {epoch + 1} / {n_epochs}')

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

with torch.no_grad():
    for x_val, y_val in val_loader:
        a_model.eval()

        yhat = a_model(x_val)


        print('Here is one')
        print(yhat)
        print(y_val)
        print()

# lets do a simply pytorch model

import numpy as np
inputs = np.array([[0, 0, 1],
                   [1, 1, 1],
                   [1, 0, 1],
                   [0, 1, 1]])

outputs = np.array([[0, 1, 1, 0]]).T


# lets create model here


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.data import TensorDataset

x_tensor = torch.from_numpy(inputs).float()
y_tensor = torch.from_numpy(outputs).float()
dataset = TensorDataset(x_tensor, y_tensor)

train_data, test_data = random_split(dataset, [3,1])
train_loader = DataLoader(train_data)
test_loader = DataLoader(test_data)


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3,3)
        self.fc2 = nn.Linear(3,3)
        self.fc3 = nn.Linear(3,1)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

model = MyModel()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


# onw can train it

for _ in range(100):

	for x, y in train_loader:
		model.train()
		y_predict = model.forward(x)
		loss = loss_fn(y, y_predict)
		loss.backward()

		optimizer.step()
		optimizer.zero_grad()

with torch.no_grad():
	for x, y in test_loader:
		model.eval()

		y_predict = model(x)
		print(y_predict)
		print(y)

for name, param in model.named_parameters():
	if param.requires_grad:
		print (name, param.data)

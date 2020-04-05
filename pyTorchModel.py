import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torch.nn.functional as F

# fix the randomness
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_train_and_test_data_loader_from_data_model(a_datamodel):
    x_tensor = torch.from_numpy(a_datamodel.X).float()
    y_tensor = torch.from_numpy(a_datamodel.y).float()
    dataset = TensorDataset(x_tensor, y_tensor)

    train_dataset, val_dataset = random_split(dataset, [129, 40])
    train_loader = DataLoader(dataset=train_dataset)  #, batch_size=10)
    val_loader = DataLoader(dataset=val_dataset)  #, batch_size=10)
    return train_loader, val_loader

# print(train_loader[0])

# lr = 0.001
# n_epochs = 100


# the model
class MyModelCnnTrend1dConvo(nn.Module):
    def __init__(self, in_channels=5, kernal_sizes=5, pool_sizes=2, weeks_to_predict=30):
        super().__init__()  # because this guys is subclass of nn.Module
        # self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        # self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        # Instead of our custom parameters, we use a Linear layer with single input and single output
        # self.linear = nn.Linear(1, 1)  # 1 feature in, 1 feature out
        self.pool = nn.MaxPool1d(pool_sizes)
        # in channel is actually number of keywords, that we use...

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=2 * in_channels, kernel_size=kernal_sizes)

        length = math.floor((weeks_to_predict - (kernal_sizes - 1)) / pool_sizes)
        # print(length)
        # each channel is a kw, in channel=number of kw
        # we do the process of convo...

        # weeks to predict = w,
        self.conv2 = nn.Conv1d(in_channels=2 * in_channels, out_channels=3 * in_channels, kernel_size=kernal_sizes)

        # now the result is 5 * (30 - (5-1) - (5-1))
        length = math.floor((length - (kernal_sizes - 1)) / pool_sizes)
        # print(length)

        # fully connected layers
        self.fc1 = nn.Linear(3 * in_channels * length, int(3 * in_channels * length / 2))
        self.fc2 = nn.Linear(int(3 * in_channels * length / 2), 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        # maybe i need to add pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))

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
def train_kit(n_epochs=100, lr=0.001):
    a_model = MyModelCnnTrend1dConvo().to(device)
    # define the optimizer

    a_loss_fn = nn.MSELoss(reduction='mean')  # nn.BCELoss()  # CrossEntropyLoss()  # nn.MSELoss(reduction='mean')
    an_optimizer = optim.SGD(a_model.parameters(), lr=lr, momentum=0.9)

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
    return a_model


def eval_kit(a_val_loader, a_model):
    total_correct_count = 0
    one_zero_pre_count = 0
    zero_one_pre_count = 0
    one_zero_correct_pre_count = 0
    zero_one_correct_pre_count = 0
    total_count = 0

    with torch.no_grad():
        for x_val, y_val in a_val_loader:
            a_model.eval()

            yhat = a_model(x_val)
            total_count += 1

            print('Here is one')
            print(yhat)
            print(y_val)

            # check if this is correct, then lets add to count
            if yhat.numpy()[0][0] > yhat.numpy()[0][1]:
                one_zero_pre_count += 1
                # print('this is 1 0')
                if y_val.numpy()[0][0] == 1:
                    total_correct_count += 1
                    one_zero_correct_pre_count += 1
            else:
                zero_one_pre_count += 1
                if y_val.numpy()[0][0] == 0:
                    total_correct_count += 1
                    zero_one_correct_pre_count += 1

            print()

    # need to write check accuracy
    print(f'total accuracy: {total_correct_count / total_count}')
    if one_zero_pre_count > 0:
        print(f'those predict 1,0: {one_zero_correct_pre_count / one_zero_pre_count}')
    if zero_one_pre_count > 0:
        print(f'those predict 0,1: {zero_one_correct_pre_count / zero_one_pre_count}')

# eval_kit(val_loader, a_model)
# accuracy with prediction 1, 0

# accuracy with prediction 0, 1

# accuracy with actual 1, 0

# accuracy with actual 0, 1

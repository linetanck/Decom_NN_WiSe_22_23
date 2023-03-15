import torch.nn as nn
import torch.optim as optim
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt



batch_size = 64
class BookClassification(nn.Module):
    def __init__(self, input_size):
        super(BookClassification, self).__init__()
        self.layer_1 = nn.Linear(input_size, 64)
        self.hidden2 = nn.Linear(64,64)
        self.layer_out = nn.Linear(64, 6)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=0.01)

        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.hidden2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x
def accuracy(y_pred, y_test):
    y_pred_tag = torch.max(y_pred, 1).indices

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]

    return acc

data = pd.read_csv('data_1/sf_token_training_set.csv')
target_data = data[['rating']]
train_data = data[['tokens']]

target_data = target_data.values
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
encoded_tokens = encoder.fit_transform(train_data).toarray()

encoded_tokens_array = np.array(encoded_tokens)
target_data_array = np.array(target_data)

train_dataset = (encoded_tokens_array, target_data_array)

tensor_x = torch.Tensor(encoded_tokens_array) # transform to torch tensor
tensor_y = torch.Tensor(target_data_array)

my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset

train_dataloader = DataLoader(dataset=my_dataset, batch_size=batch_size, shuffle=True)

num_features = 21395
model = BookClassification(num_features)
device = 'cpu'
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)
model_saved_name = "book_classification_lr_0.001_bs100_ep_5_plot"

epochs = 10
epoch_loss_list = []
epoch_acc_list = []

model.train()
for e in range(1, epochs + 1):
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        y_pred = model(X_batch)
        y_batch = y_batch.to(int)
        y_batch = y_batch.squeeze()

        loss = criterion(y_pred, y_batch)
        acc = accuracy(y_pred, y_batch)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc

        epoch_loss_list.append(loss.item())
        epoch_acc_list.append(acc)

    print(
        f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_dataloader):.5f} | Acc: {epoch_acc / len(train_dataloader):.3f}')
    plt.plot(epoch_acc_list)
    plt.show()
    plt.plot(epoch_loss_list)
    plt.show()
    torch.save(model.state_dict(), 'data_1/' + model_saved_name + '.pt')

import torch.nn as nn
import torch.optim as optim
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt



batch_size = 100
class BookClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()
        self.layer_1 = nn.Linear(21395, 16384)
        self.hidden2 = nn.Linear(16384, 8000)
        self.hidden3 = nn.Linear(8000, 2000)
        self.hidden4 = nn.Linear(2000, 100)
        self.hidden6 = nn.Linear(100, 64)
        self.layer_out = nn.Linear(64, 6)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.hidden2(x))
        x = self.relu(self.hidden3(x))
        x = self.relu(self.hidden4(x))
        x = self.relu(self.hidden6(x))
        x = self.layer_out(x)

        return x


data = pd.read_csv('data/tokenized/sf_token_training_set.csv')
for i in data["rating"]:
    j = int(i)
    j = round(j)
    data['rating'] = data['rating'].replace(to_replace=i, value=j)

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

model = BookClassification()

device = 'cpu'
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
model_saved_name = "book_classification_lr_0.001_bs100_ep_5_plot_2"

epochs = 5
epoch_loss_list = []
epoch_accuracy_list = []


for epoch in range(epochs):  # loop through epoch
    epoch_loss = 0
    epoch_accuracy = 0
    batch = 0
    for data, label in train_dataloader:  # iterating through the train loader and using every batch it can get
        model.train() # setting model in train mode
        data = data.to(device)
        label = label.to(device)
        batch += 1  # sending things to cpu

        output = model(data)
        reduced = torch.max(output, 1)
        label = label.to(int)
        label = label.squeeze()
        loss = criterion(output, label)

        loss.backward()  # calculates where the changes in the weights and biases should be
        optimizer.step()

        acc = ((output.argmax(dim=1) == label).float().mean()).item()
        epoch_accuracy += acc / len(train_dataloader)  # adds all accuracies for an epoch together
        epoch_loss += loss.item() / len(train_dataloader)

        print(f"batch {int(batch)}; "f"batch loss: {loss:.3f} ")
        print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch + 1, epoch_accuracy, epoch_loss))

        epoch_accuracy_list.append(epoch_accuracy)
        epoch_loss_list.append(epoch_loss)

    plt.plot(epoch_accuracy_list)
    #plt.savefig("graph_Test_accuracy.png")
    plt.show()
    plt.plot(epoch_loss_list)
    #plt.savefig("graph_Test_loss.png")
    plt.show()
    torch.save(model.state_dict(), 'data/' + model_saved_name + '.pt')


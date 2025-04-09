import torch
import torch.optim as optim
import torch.utils.data as Data
import torch.nn as nn
import os
import numpy as np


from denoise_fre_fc import Denoise_net
from tqdm import tqdm
import torch.nn.functional as F


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


BATCH_SIZE = 1000
learning_rate = 1e-4
epochs = 50
mini_loss = 1

model_name = 'Denoise_net'
model = Denoise_net()

loss1 = nn.MSELoss(reduction='mean')



raw_eeg = np.load('dataset/train_input.npy')
clean_eeg = np.load('dataset/train_output.npy')


test_input = np.load('dataset/test_input.npy')
test_output = np.load('dataset/test_output.npy')

raw_eeg = raw_eeg[0:1000]
clean_eeg = clean_eeg[0:1000]
test_input = test_input[0:1000]
test_output = test_output[0:1000]

train_input = torch.from_numpy(raw_eeg)
train_output = torch.from_numpy(clean_eeg)

train_torch_dataset = Data.TensorDataset(train_input,  train_output)

train_loader = Data.DataLoader(
    dataset=train_torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_input = torch.from_numpy(test_input)
test_output = torch.from_numpy(test_output)


test_torch_dataset = Data.TensorDataset(test_input, test_output)

test_loader = Data.DataLoader(
    dataset=test_torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("torch.cuda.is_available() = ", torch.cuda.is_available())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(epochs):

    train_acc = 0
    train_loss = 0

    total_train_loss_per_epoch = 0
    average_train_loss_per_epoch = 0
    train_step_num = 0

    for (train_input, train_output) in tqdm (train_loader):
        model.train()
        train_loss_list = []
        train_input = train_input.float().to(device)
        train_output = train_output.float().to(device)
        optimizer.zero_grad()

        train_preds = model(train_input)
        train_loss1 = loss1(train_preds, train_output)
        train_loss1.backward()
        train_loss_list.append(train_loss1.cpu().detach().numpy())
        optimizer.step()

    for (test_input,  test_output) in  test_loader:
        model.eval()
        test_loss_list = []
        test_input = test_input.float().to(device)
        test_output = test_output.float().to(device)

        test_preds = model(test_input)

        test_loss = loss1(test_preds, test_output)

        test_loss_list.append(test_loss.cpu().detach().numpy())

    tqdm.write(f'epoch:{epoch} train_loss:{np.mean(train_loss_list):.8f} test_loss:{np.mean(test_loss_list):.8f}')

    if np.mean(test_loss_list) < mini_loss:
        print('save model')
        mini_loss = np.mean(test_loss_list)
        torch.save(model.state_dict(), f'zvanills_model_{epoch}_{mini_loss:.3f}.pkl')
    



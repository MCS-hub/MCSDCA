import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from cmapss_process import preprocess
from torch.utils.data import TensorDataset, DataLoader
import models
import xlsxwriter

torch.manual_seed(42)
# check gpu
cuda_available = False
if torch.cuda.is_available():
    cuda_available = True

# get data

dataset = 'FD001'

remaining_sensors = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17',
                     's_20', 's_21']
epochs = 40

model_style = 'LSTM'
model_name = 'cmapsslstm'
sequence_length = 23
exp_smooth = 0.1
train_array, train_label, test_array, test_label, train_split_array, train_split_label, val_split_array, val_split_label = preprocess(
    dataset, remaining_sensors, sequence_length, exp_smooth=exp_smooth, style=model_style)

train_dataset = TensorDataset(train_array, train_label)
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

activation = None
dropout = 0.5

model_config = {'dropout': 0.3, 'num_layers': 1, 'input_sz': 14, 'hidden_size': 128, 'cuda': cuda_available}

model = getattr(models, model_name)(model_config)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.MSELoss()

if cuda_available:
    model.to('cuda')
    loss_function.to('cuda')

# train
model.train()
for epoch in range(epochs):
    for step, (x_batch, y_batch) in enumerate(train_dataloader):
        print('step: ', step)
        if cuda_available:
            x_batch = x_batch.to('cuda')
            y_batch = y_batch.to('cuda')

        model.zero_grad()
        y_hat = model(x_batch)
        y_hat = torch.squeeze(y_hat)
        loss = loss_function(y_hat, y_batch)
        loss.backward()
        optimizer.step()

model.eval()
if cuda_available:
    test_array = test_array.to('cuda')
    test_label = test_label.to('cuda')
y_hat_test = model(test_array)
y_hat_test = torch.squeeze(y_hat_test)
test_loss = loss_function(y_hat_test, test_label)
test_rmse = np.sqrt(test_loss.detach().numpy())
print('test rmse:', test_rmse)

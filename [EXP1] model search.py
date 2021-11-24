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
# get data
dataset_list = ['FD002','FD004']
max_sequence_length = {'FD001': 30, 'FD002': 20, 'FD003': 38, 'FD004': 19}

cuda_available = False
if torch.cuda.is_available():
    cuda_available = True

for dataset in dataset_list:

    remaining_sensors = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20',
                         's_21']

    sequence_length_list = [i for i in range(5, max_sequence_length[dataset])]
    batch_size_list = [32, 64]
    alpha_list = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    activation_list = ['Tanh', 'Sigmoid']
    model_list = ['FNN', 'RNN', 'LSTM']
    epochs = 40

    FNN_unit_list = [8, 16, 32, 64, 128, 256, 512, 1024]

    LSTM_unit_list = [16, 32, 64, 128, 256]

    RNN_unit_list = [16, 32, 64, 128, 256]

    dropout_list = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    ITERATIONS = 200

    import random
    # Workbook is created
    workbook = xlsxwriter.Workbook(dataset+'.xlsx')
    worksheet = workbook.add_worksheet()
    # add_sheet is used to create sheet.
    worksheet.write(0,0,'model_style')
    worksheet.write(0,1,'sequence_length')
    worksheet.write(0,2,'exp_smooth')
    worksheet.write(0,3,'batch_size')
    worksheet.write(0,4,'model_config')
    worksheet.write(0,5,'test_rmse')

    for iteration in range(ITERATIONS):
        model_style = random.sample(model_list, 1)[0]
        sequence_length = random.sample(sequence_length_list, 1)[0]
        exp_smooth = random.sample(alpha_list, 1)[0]
        train_array, train_label, test_array, test_label, train_split_array, train_split_label, val_split_array, val_split_label = preprocess(
            dataset, remaining_sensors, sequence_length, exp_smooth=exp_smooth, style=model_style)

        train_dataset = TensorDataset(train_array, train_label)
        batch_size = random.sample(batch_size_list, 1)[0]
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

        activation = random.sample(activation_list, 1)[0]
        dropout = random.sample(dropout_list, 1)[0]

        if model_style == 'FNN':
            model_name = 'cmapssfnn'
            FNN_layer = 3
            FNN_unit = random.sample(FNN_unit_list[0:len(FNN_unit_list) + 1 - FNN_layer], 1)[0]
            structure = list()
            for i in range(FNN_layer):
                structure.append((2 ** i) * FNN_unit)
            input_sz = len(remaining_sensors)*sequence_length
            model_config = {'activation': activation, 'dropout': dropout, 'structure': structure, 'input_sz': input_sz, 'cuda': cuda_available}
        elif model_style == 'LSTM':
            model_name = 'cmapsslstm'
            LSTM_unit = random.sample(LSTM_unit_list, 1)[0]
            input_sz = len(remaining_sensors)
            model_config = {'dropout': dropout, 'num_layers': 1, 'input_sz': input_sz, 'hidden_size': LSTM_unit, 'cuda': cuda_available}
        elif model_style == 'RNN':
            model_name = 'cmapssrnn'
            RNN_unit = random.sample(RNN_unit_list, 1)[0]
            input_sz = len(remaining_sensors)
            model_config = {'dropout': dropout, 'num_layers': 1, 'input_sz': input_sz, 'hidden_size': RNN_unit, 'cuda': cuda_available}


        model = getattr(models,model_name)(model_config)
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
        test_rmse = np.sqrt(test_loss.detach().cpu().numpy())
        print('test rmse:', test_rmse)
    
        worksheet.write(iteration+1, 0, model_style)
        worksheet.write(iteration+1, 1, sequence_length)
        worksheet.write(iteration+1, 2, exp_smooth)
        worksheet.write(iteration+1, 3, batch_size)
        worksheet.write(iteration+1, 4, str(model_config))
        worksheet.write(iteration+1, 5, test_rmse)

    workbook.close()
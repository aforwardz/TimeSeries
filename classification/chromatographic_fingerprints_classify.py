import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import fastai
from fastai.vision.all import *
pd.set_option("display.precision", 10)


DATASET_DIR = "./dataset/图神经网络/"


def preprocess_data():
    for r, _, files in os.walk(DATASET_DIR):
        print(r, files)
        if r != DATASET_DIR:
            for file in files:
                if file.endswith('txt'):
                    print(r, file)
                    get_df_from_file(os.path.join(r, file))


def get_df_from_file(fp):
    t1, t2 = "指纹图谱采样信号数据", "指纹图谱积分数据"
    data_list = []
    with open(fp, 'r', encoding='gbk') as f:
        process = False
        for line in f:
            # print(line.strip())
            line = line.strip()
            if t1 in line:
                process = True
                continue
            if t2 in line:
                break
            if process:
                values = line.split()
                # print(values)
                data_list.append([values[0], values[-1]])

    df = pd.DataFrame(data_list[1:], columns=['time', 'value'])
    df['time'] = df['time'].astype(float)
    df['value'] = df['value'].astype(float)
    df = df.set_index(['time'])
    print(df)

    df.to_csv(fp.replace('txt', 'csv'))


def check_data():
    for r, _, files in os.walk(DATASET_DIR):
        print(r, files)
        if r != DATASET_DIR:
            for file in files:
                if file.endswith('txt'):
                    print(r, file)
                    fp = os.path.join(r, file)
                    df_pf = fp.replace('txt', 'csv')
                    if not os.path.exists(df_pf):
                        raise OSError('%s Not Exists' % df_pf.split('/')[-1])
                    else:
                        df = pd.read_csv(df_pf, index_col=['time'])
                        fn = df_pf.split('/')[-1]
                        if (fn.startswith('SC') or fn.startswith('TC')) and (len(df) != 16800 and len(df) != 16801):
                            raise ValueError('%s Len: %d' % (fn, len(df)))


class BlockLSTM(nn.Module):
    def __init__(self, time_steps, num_layers, lstm_hs, dropout=0.8, attention=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size=time_steps, hidden_size=lstm_hs, num_layers=num_layers)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # input is of the form (batch_size, num_layers, time_steps), e.g. (128, 1, 512)
        x = torch.transpose(x, 0, 1)
        # lstm layer is of the form (num_layers, batch_size, time_steps)
        x, (h_n, c_n) = self.lstm(x)
        # dropout layer input shape (Sequence Length, Batch Size, Hidden Size * Num Directions)
        y = self.dropout(x)
        # output shape is same as Dropout intput
        return y


class BlockFCNConv(nn.Module):
    def __init__(self, in_channel=1, out_channel=128, kernel_size=8, momentum=0.99, epsilon=0.001, squeeze=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size)
        self.batch_norm = nn.BatchNorm1d(num_features=out_channel, eps=epsilon, momentum=momentum)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input (batch_size, num_variables, time_steps), e.g. (128, 1, 512)
        x = self.conv(x)
        # input (batch_size, out_channel, L_out)
        x = self.batch_norm(x)
        # same shape as input
        y = self.relu(x)
        return y


class BlockFCN(nn.Module):
    def __init__(self, time_steps, channels=[1, 128, 256, 128], kernels=[8, 5, 3], mom=0.99, eps=0.001):
        super().__init__()
        self.conv1 = BlockFCNConv(channels[0], channels[1], kernels[0], momentum=mom, epsilon=eps, squeeze=True)
        self.conv2 = BlockFCNConv(channels[1], channels[2], kernels[1], momentum=mom, epsilon=eps, squeeze=True)
        self.conv3 = BlockFCNConv(channels[2], channels[3], kernels[2], momentum=mom, epsilon=eps)
        output_size = time_steps - sum(kernels) + len(kernels)
        self.global_pooling = nn.AvgPool1d(kernel_size=output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # apply Global Average Pooling 1D
        y = self.global_pooling(x)
        return y


class LSTMFCN(nn.Module):
    def __init__(self, time_steps, num_variables=1, lstm_hs=256, channels=[1, 128, 256, 128]):
        super().__init__()
        self.lstm_block = BlockLSTM(time_steps, 1, lstm_hs)
        self.fcn_block = BlockFCN(time_steps)
        self.dense = nn.Linear(channels[-1] + lstm_hs, num_variables)
        self.softmax = nn.LogSoftmax(dim=1) #nn.Softmax(dim=1)

    def forward(self, x):
        # input is (batch_size, time_steps), it has to be (batch_size, 1, time_steps)
        x = x.unsqueeze(1)
        # pass input through LSTM block
        x1 = self.lstm_block(x)
        x1 = torch.squeeze(x1)
        # pass input through FCN block
        x2 = self.fcn_block(x)
        x2 = torch.squeeze(x2)
        # concatenate blocks output
        x = torch.cat([x1, x2], 1)
        # pass through Linear layer
        x = self.dense(x)
        #x = torch.squeeze(x)
        # pass through Softmax activation
        y = self.softmax(x)
        return y


class SimpleLearner():
    def __init__(self, data, model, loss_func, wd=1e-5):
        self.data, self.model, self.loss_func = data, model, loss_func
        self.wd = wd

    def update_manualgrad(self, x, y, lr):
        y_hat = self.model(x)
        # weight decay
        w2 = 0.
        for p in self.model.parameters():
            w2 += (p ** 2).sum()
        # add to regular loss
        loss = self.loss_func(y_hat, y) + w2 * self.wd
        loss.backward()
        with torch.no_grad():
            for p in self.model.parameters():
                p.sub_(lr * p.grad)
                p.grad.zero_()
        return loss.item()

    def update(self, x, y, lr):
        opt = optim.Adam(self.model.parameters(), lr)
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        return loss.item()

    def fit(self, epochs=1, lr=1e-3):
        """Train the model"""
        losses = []
        for i in tqdm(range(epochs)):
            for x, y in self.data[0]:
                current_loss = self.update(x, y, lr)
                losses.append(current_loss)
        return losses

    def evaluate(self, X):
        """Evaluate the given data loader on the model and return predictions"""
        result = None
        for x, y in X:
            y_hat = self.model(x)
            y_hat = y_hat.cpu().detach().numpy()
            result = y_hat if result is None else np.concatenate((result, y_hat), axis=0)
        return result


def one_hot_encode(input, labels):
    m = input.shape[0]
    output = np.zeros((m, labels), dtype=int)
    row_index = np.arange(m)
    output[row_index, input] = 1
    return output


def split_xy(data, classes):
    X = data[:, 1:]
    y = data[:, 0].astype(int)
    # hot encode
    # y = one_hot_encode(y, classes)
    return X, y


def create_dataset(X, y, device):
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)
    return TensorDataset(X_tensor, y_tensor)


def load_data(path, classes):
    data = np.loadtxt(path)
    return split_xy(data, classes)


def train(fast=False):
    classes = 2
    # load training dataset
    X_train, y_train = load_data('./dataset/Earthquakes_TRAIN.txt', classes=classes)
    # load testing dataset
    X_test, y_test = load_data('./dataset/Earthquakes_TEST.txt', classes=classes)
    print('X_train %s   y_train %s' % (X_train.shape, y_train.shape))
    print('X_test  %s   y_test  %s' % (X_test.shape, y_test.shape))

    train_ds = create_dataset(X_train, y_train, 'cpu')
    test_ds = create_dataset(X_test, y_test, 'cpu')

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=False)  # , sampler = sampler)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)

    time_steps = X_train.shape[1]
    num_variables = classes

    model = LSTMFCN(time_steps, num_variables)

    # # model summary
    # for m in model.children():
    #     print(m.training)  # , m)
    #     for j in m.children():
    #         print(j.training, j)

    loss_func = nn.NLLLoss()  # weight=weights
    # acc_func = accuracy_thresh
    lr = 3e-3
    bs = 64

    if fast:
        data = DataLoader(train_dl=train_dl, valid_dl=test_dl, batch_size=64, shuffle=False)
        learner = Learner(data, model, loss_func=loss_func, lr=lr)
        learner.fit(10, lr=3e-3)
        y_preds = learner.get_preds(learner.dls.valid)
        print(y_preds)
    else:
        learner = SimpleLearner([train_dl, test_dl], model, loss_func)
        losses = learner.fit(10, lr=lr)

        plt.plot(losses)

        y_pred = learner.evaluate(test_dl)
        print(y_pred)
        pred_loss = ((y_test - y_pred.argmax(axis=1)) ** 2).mean()
        print(pred_loss)


if __name__ == '__main__':
    # get_df_from_file(os.path.join(DATASET_DIR, '训练集SC-57/SD-2.txt'))
    # preprocess_data()
    # check_data()
    train(fast=True)




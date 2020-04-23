# modified from https://blog.floydhub.com/gru-with-pytorch/
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import gc
import torch
from sklearn import metrics

lr = 0.01
model_type = 'GRU'


def get_device(num):
    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda:{}".format(int(num)))
    else:
        device = torch.device("cpu")
    return device


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=1, n_layers=2, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.drop_prob = drop_prob
        self.device = get_device(0)
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True, dropout=self.drop_prob)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.relu = nn.ReLU()

    def change_device(self, num):
        self.device = get_device(num)

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        return hidden


class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = get_device(0)
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.lstm(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device))
        return hidden


def train(train_loader, learn_rate, hidden_dim=256, EPOCHS=5, model_type="GRU"):
    # Setting common hyperparameters
    input_dim = next(iter(train_loader))[0].shape[-1]
    output_dim = 1
    n_layers = 2
    model_type = model_type
    EPOCHS = EPOCHS
    hidden_dim = hidden_dim
    learn_rate = learn_rate
    device = get_device(0)
    # Instantiating the models
    if model_type == "GRU":
        model = GRUNet(input_dim, hidden_dim, output_dim, n_layers)
    else:
        model = LSTMNet(input_dim, hidden_dim, output_dim, n_layers)
    # model = nn.DataParallel(model)
    model.to(device)

    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    model.train()
    print("Starting Training of {} model".format(model_type))
    epoch_times = []
    # Start training loop
    for epoch in range(1, EPOCHS + 1):
        start_time = time.clock()
        h = model.init_hidden(batch_size)
        avg_loss = 0.
        counter = 0
        for x, label in train_loader:
            counter += 1
            if model_type == "GRU":
                h = h.data
            else:
                h = tuple([e.data for e in h])
            model.zero_grad()

            out, h = model(x.to(device).float(), h)
            loss = criterion(out, label.to(device).float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            if counter % 200 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter,
                                                                                           len(train_loader),
                                                                                           avg_loss / counter))
        current_time = time.clock()
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss / len(train_loader)))
        print("Total Time Elapsed: {} seconds".format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    return model


def evaluate(model, test_x, test_y, test_loader, model_type="GRU"):
    model.eval()
    outputs = []
    targets = []
    start_time = time.clock()
    inp = torch.from_numpy(np.array(test_x))
    labs = torch.from_numpy(np.array(test_y))
    model.change_device(1)
    h = model.init_hidden(batch_size)
    for x, label in test_loader:
        out, h = model(x.to(get_device(1)).float(), h)
        if model_type == "GRU":
            h = h.data
        else:
            h = tuple([e.data for e in h])
        # outputs.append(label_sc.inverse_transform(out.cpu().detach().numpy()).reshape(-1).flatten().tolist())
        # targets.append(label_sc.inverse_transform(label.numpy()).reshape(-1).flatten().tolist())
        outputs += out.cpu().detach().numpy().flatten().tolist()
        targets += label.flatten().tolist()
    print("Evaluation Time: {}".format(str(time.clock() - start_time)))
    sMAPE = 0
    outputs = np.array(outputs)
    targets = np.array(targets)
    for i in range(len(outputs)):
        sMAPE += np.mean(abs(outputs[i] - targets[i]) / (targets[i] + outputs[i]) / 2) / len(outputs)

    print("sMAPE: {}%".format(sMAPE * 100))

    return outputs, targets, sMAPE


def eval_metric(outputs, targets):
    outputs = np.around(outputs)
    precision = metrics.precision_score(targets, outputs)
    recall = metrics.recall_score(targets, outputs)
    f1 = metrics.f1_score(targets, outputs)
    fpr, tpr, thres = metrics.roc_curve(targets, outputs)
    print('precision: ', precision)
    print('recall: ', recall)
    print('f1-score: ', f1)
    print('fpr: ', fpr)
    print('tpr', tpr)
    df = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Threshold': thres})
    df.to_csv('eval.csv', index=False)
    draw_roc(tpr, fpr)


def draw_roc(tpr, fpr):
    fpr = dict()
    tpr = dict()
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC {}'.format(model_type))
    plt.legend(loc="lower right")
    plt.savefig('ROC.jpg')


if __name__ == '__main__':
    batch_size = 1024

    df = pd.read_csv('finally_preprocessed.csv', header=None, dtype=np.float)

    # Scaling the input data
    sc = MinMaxScaler()
    label_sc = MinMaxScaler()
    data = sc.fit_transform(df.values)
    # Obtaining the Scale for the labels(usage data) so that output can be re-scaled to actual value during evaluation
    label_sc.fit(df.iloc[:, 0].values.reshape(-1, 1))

    # df = pd.read_csv('cracked.csv', header=None, dtype=np.float)
    print('tot len: ', df.shape[0])
    tmp = df.iloc[:, -1]
    tmp = np.array(tmp)
    tmp[tmp != 0] = 1
    df.iloc[:, -1] = tmp
    len_http = 248294
    len_ack = 313462
    len_udp = 1187114

    train_http = int(0.8 * len_http)
    train_ack = int(0.8 * len_ack)
    train_udp = int(0.8 * len_udp)

    train_range = list(range(train_http)) + list(range(len_http, len_http + train_ack)) + list(
        range(len_http + len_ack, len_http + len_ack + train_udp))
    test_range = list(range(train_http, len_http)) + list(range(len_http + train_ack, len_http + len_ack)) + list(
        range(len_http + len_ack + train_udp, len_http + len_ack + len_udp))

    # train_x = df.iloc[train_range, :-1]
    # train_y = df.iloc[train_range, -1]
    #
    # test_x = df.iloc[test_range, :-1]
    # test_y = df.iloc[test_range, -1]
    train_set = df.iloc[train_range, :].values
    test_set = df.iloc[test_range, :].values

    # print('train len: ', train_x.shape[0])
    # print('test len: ', test_x.shape[0])
    # Define lookback period and split inputs/labels
    lookback = 10
    sample_portion = .5

    train_x = np.zeros((len(train_set) - lookback, lookback, df.shape[1]))
    train_y = np.zeros(len(train_set) - lookback)

    for i in range(lookback, len(train_set)):
        train_x[i - lookback] = train_set[i - lookback:i]
        train_y[i - lookback] = train_set[i, -1]
    train_x = train_x.reshape(-1, lookback, train_set.shape[1])
    train_y = train_y.reshape(-1, 1)

    test_x = np.zeros((len(test_set) - lookback, lookback, df.shape[1]))
    tests_y = np.zeros(len(test_set) - lookback)

    for i in range(lookback, len(test_set)):
        test_x[i - lookback] = test_set[i - lookback:i]
        tests_y[i - lookback] = test_set[i, -1]
    test_x = test_x.reshape(-1, lookback, test_set.shape[1])
    test_y = tests_y.reshape(-1, 1)
    ########
    # train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    # train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    #
    # gru_model = train(train_loader, lr, EPOCHS=5, model_type=model_type)
    #
    # torch.save(gru_model.state_dict(), './trained_model.pth')
    ############

    gru_model = GRUNet(test_x.shape[-1])
    gru_model.change_device(1)
    gru_model.load_state_dict(torch.load('./trained_model.pth', map_location=get_device(1)))
    gru_model.to(get_device(1))

    # del train_loader
    # del train_data
    # del train_set
    # del train_x
    # del train_y
    # gc.collect()
    torch.cuda.empty_cache()
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, drop_last=True)

    outputs, target, _ = evaluate(gru_model, test_x, test_y, test_loader)
    eval_metric(outputs, target)

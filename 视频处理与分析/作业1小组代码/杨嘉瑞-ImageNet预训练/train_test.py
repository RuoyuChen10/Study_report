import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from MyDataset import MyDataset
from model import CNNModel
from sklearn.metrics import hamming_loss
import numpy as np
import matplotlib.pyplot as plt
from pretrain_model import get_pretrain_model


def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    error_rate = 0
    correct_all_num = 0
    correct_pred_num = 0
    pred_all_num = 0
    for batch_index, (data, target, _) in enumerate(test_loader):
        data, target = Variable(data).to(device),Variable(target).to(device)
        output = torch.sigmoid(model(data))
        test_loss += criterion(output, target).item()
        target = target.squeeze().detach().cpu().numpy()
        pred = np.round_(output.squeeze().detach().cpu().numpy())
        error_rate += hamming_loss(pred, target)
        correct_all_num += sum(target == 1)
        correct_pred_num += sum((target == 1) & (pred == 1))
        pred_all_num += sum(pred == 1)
    test_loss /= len(test_loader)
    error_rate /= len(test_loader)
    precision = 0 if pred_all_num == 0 else correct_pred_num / pred_all_num
    recall = 0 if correct_all_num == 0 else correct_pred_num / correct_all_num
    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return test_loss, error_rate, precision, recall, f1


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    count = 0
    total_loss = 0
    for batch_index, (data, target, _) in enumerate(train_loader):
        data, target = Variable(data).to(device), Variable(target).to(device)
        optimizer.zero_grad()
        output = torch.sigmoid(model(data))
        loss = criterion(output, target)
        count += 1
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    total_loss /= count
    return total_loss


def main_1(nepoch, batch_size, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_datasets = MyDataset(train_path)
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=8)
    test_datasets = MyDataset(test_path)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=1, num_workers=0)
    criterion = nn.BCELoss()
    model = model.to(device)
    model_name = 'model.pth'
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimal = 0
    for epoch in range(nepoch):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        test_loss, error_rate, precision, recall, f1 = test(model, test_loader, criterion, device)
        train_loss_set.append(train_loss)
        test_loss_set.append(test_loss)
        error_rate_set.append(error_rate)
        precision_set.append(precision)
        recall_set.append(recall)
        f1_set.append(f1)
        print(
            'Epoch: {} --- Train Loss: {:.6f} --- Test Loss: {:.6f} --- Test Error Rate: {:.3f} --- Precision: {:.3f} --- Recall: {:.3f} --- F1 Value: {:.3f}'.format(
                epoch, train_loss, test_loss, error_rate, precision, recall, f1))
        if f1 - error_rate > optimal:
            torch.save(model.state_dict(), '{0}/{1}'.format(model_path, model_name))
            optimal = f1 - error_rate
    x = np.linspace(0, nepoch, nepoch)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, train_loss_set, color='yellow', label="Train Loss")
    ax.plot(x, test_loss_set, color='red', label="Test Loss")
    plt.legend()
    # plt.ylim(0)
    plt.xlabel('Epoch', fontsize=10, weight='bold')
    plt.savefig('loss.png', dpi=600)
    plt.show()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, error_rate_set, color='green', label="Error Rate")
    ax.plot(x, precision_set, color='red', label="Precision")
    ax.plot(x, recall_set, color='blue', label="Recall")
    ax.plot(x, f1_set, color='purple', label="F1 Value")
    plt.legend()
    # plt.ylim(0)
    plt.xlabel('Epoch', fontsize=10, weight='bold')
    plt.savefig('measure.png', dpi=600)
    plt.show()


def main_2(nepoch, batch_size, mode):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for fold in range(10):
        if mode == 0:
            model = CNNModel()
        else:
            model = get_pretrain_model(type=1)
        train_datasets = MyDataset(root_path, flag=0, fold=fold)
        train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=8)
        test_datasets = MyDataset(root_path, flag=1, fold=fold)
        test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=1, num_workers=0)
        criterion = nn.BCELoss()
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        for epoch in range(nepoch):
            train_loss = train(model, train_loader, optimizer, criterion, device)
            test_loss, error_rate, precision, recall, f1 = test(model, test_loader, criterion, device)
            if fold == 0:
                train_loss_set.append(0.1 * train_loss)
                test_loss_set.append(0.1 * test_loss)
                error_rate_set.append(0.1 * error_rate)
                precision_set.append(0.1 * precision)
                recall_set.append(0.1 * recall)
                f1_set.append(0.1 * f1)
            else:
                train_loss_set[epoch] += 0.1 * train_loss
                test_loss_set[epoch] += 0.1 * test_loss
                error_rate_set[epoch] += 0.1 * error_rate
                precision_set[epoch] += 0.1 * precision
                recall_set[epoch] += 0.1 * recall
                f1_set[epoch] += 0.1 * f1
        print('finish training fold {}'.format(fold + 1))
    x = np.linspace(0, nepoch, nepoch)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, train_loss_set, color='yellow', label="Train Loss")
    ax.plot(x, test_loss_set, color='red', label="Test Loss")
    plt.legend()
    # plt.ylim(0)
    plt.xlabel('Epoch', fontsize=10, weight='bold')
    plt.savefig('loss.png', dpi=600)
    plt.show()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, error_rate_set, color='green', label="Error Rate")
    ax.plot(x, precision_set, color='red', label="Precision")
    ax.plot(x, recall_set, color='blue', label="Recall")
    ax.plot(x, f1_set, color='purple', label="F1 Value")
    plt.legend()
    # plt.ylim(0)
    plt.xlabel('Epoch', fontsize=10, weight='bold')
    plt.savefig('measure.png', dpi=600)
    plt.show()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train_path = 'train_data'
    test_path = 'test_data'
    root_path = '作业一数据/课程实验数据'
    model_path = 'checkpoints'
    batch_size = 8
    nepoch = 30
    mode = 0
    flag = 1
    train_loss_set, test_loss_set, error_rate_set, precision_set, recall_set, f1_set = [], [], [], [], [], []
    if flag is None:
        if mode == 0:
            main_1(nepoch, batch_size, CNNModel())
        else:
            pretrain_model = get_pretrain_model(type=1)
            main_1(nepoch, batch_size, pretrain_model)
    else:
        main_2(nepoch, batch_size, mode)

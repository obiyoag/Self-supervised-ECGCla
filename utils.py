import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
import torch
import torch.nn.functional as F


def getGenerators(dataset, batch_size, setSplit, random_seed):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(setSplit * dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_indices, test_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    trainNum = len(train_sampler)
    testNum = len(test_sampler)

    train_loader = DataLoader(dataset, batch_size, sampler=train_sampler, num_workers=8, pin_memory=True)
    test_loader = DataLoader(dataset, batch_size, sampler=test_sampler, num_workers=8, pin_memory=True)

    return train_loader, test_loader


def pre_train(model, device, train_loader, optimizer, epoch):
    model.train()
    trainNum = len(train_loader.dataset)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data).view_as(target)
        loss = mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if(batch_idx + 1) % 5 == 0:
            trainedNum = (batch_idx + 1) * len(data)
            print('Train Epoch: {} [{}/{}]\tMSE: {:.6f}'.format(epoch, trainedNum, trainNum, loss))


def pre_validation(model, device, val_loader):
    model.eval()
    valBatchNum = len(val_loader)
    mse = 0.
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data).view_as(target)
            mse += mse_loss(output, target)

    mse /= valBatchNum
    print('Validation MSE: {:.6f}'.format(mse))


def tune_train(model, device, train_loader, optimizer, epoch):
    model.train()
    trainNum = len(train_loader.sampler)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if(batch_idx + 1) % 5 == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, (batch_idx + 1) * len(data), trainNum, loss.item()))


def tune_test(model, device, test_loader):
    model.eval()
    testNum = len(test_loader.sampler)
    test_loss = 0.
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= testNum
    print('\nTest loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, testNum, 100. * correct / testNum))

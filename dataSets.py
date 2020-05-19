import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
# import matplotlib.pyplot as plt


def normalization_processing(data):

    data_mean = data.mean()
    data_std = data.std()

    data = data - data_mean
    data = data / data_std

    return data


# snr控制信噪比， 设置random_seed保证每次加噪声的随机性一样
def wgn(x, snr, random_seed):
    np.random.seed(random_seed)
    Ps = np.sum(abs(x)**2) / len(x)
    Pn = Ps / (10**((snr / 10)))
    noise = np.random.randn(len(x)) * np.sqrt(Pn)
    signal_add_noise = x + noise
    return signal_add_noise


class OriSet(Dataset):
    def __init__(self, filePath, snr):
        print("data preparing")
        self.df = pd.read_csv(filePath, header=None)  # 取出除第一行外的数据列
        self.data = self.df.iloc[:, 1:].values
        # 数据标准化和加噪声
        for id in range(len(self.data)):
            self.data[id] = normalization_processing(self.data[id])

            if snr is not None:
                self.data[id] = wgn(self.data[id], snr, id + 1)

        self.label = self.df.iloc[:, 0].copy()
        for id in range(len(self.df)):
            label = self.label.iloc[id]
            if label == 'N':
                self.label.iloc[id] = 0
            elif label == 'S':
                self.label.iloc[id] = 1
            elif label == 'V':
                self.label.iloc[id] = 2
            elif label == 'F':
                self.label.iloc[id] = 3
        print("data preparing finished")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, id):
        data = torch.tensor(self.data[id]).view(1, -1).float()
        label = torch.tensor(self.label[id]).long()
        return data, label


class PreSet(Dataset):

    def __init__(self, filePath, maskFrom, maskTo, snr):

        print("data preparing")

        self.data = pd.read_csv(filePath, header=None).iloc[:, 1:].values  # 取出除第一列外的数据列
        # 数据标准化和加噪声
        for id in range(len(self.data)):
            if snr is not None:
                self.data[id] = wgn(self.data[id], snr, id + 1)
            self.data[id] = normalization_processing(self.data[id])

        # mask数据
        self.maskData = self.data.copy()
        for id in range(len(self.maskData)):
            self.maskData[id, maskFrom:maskTo] = 0.

        print("data preparing finished")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        maskData = torch.tensor(self.maskData[id]).view(1, -1).float()
        data = torch.tensor(self.data[id]).view(1, -1).float()
        if maskData.equal(data):
            print("Oh no")
        return maskData, data


class Scratch(Dataset):
    def __init__(self, filePath):
        print("data preparing")
        self.df = pd.read_csv(filePath, header=None)  # 取出除第一行外的数据列
        self.data = self.df.iloc[:, 1:].values
        # 数据标准化和加噪声
        for id in range(len(self.data)):
            self.data[id] = normalization_processing(self.data[id])

        self.label = self.df.iloc[:, 0].copy()
        for id in range(len(self.df)):
            label = self.label.iloc[id]
            if label == 'N':
                self.label.iloc[id] = 0
            elif label == 'S':
                self.label.iloc[id] = 1
            elif label == 'V':
                self.label.iloc[id] = 2
            elif label == 'F':
                self.label.iloc[id] = 3
        print("data preparing finished")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, id):
        data = torch.randn(1, 250).float()
        label = torch.tensor(self.label[id]).long()
        return data, label
# noise visualization
# oriset = OriSet('ECGCla/data/Classified/trainSet.csv', snr=0)
# data = oriset[1][0][0]
# print(data)
# plt.plot(range(len(data)), data)
# plt.show()

# oriset = OriSet('ECGCla/data/Classified/trainSet.csv', snr=0)
# data = oriset[1][0][0]
# print(data)
# plt.plot(range(len(data)), data)
# plt.show()

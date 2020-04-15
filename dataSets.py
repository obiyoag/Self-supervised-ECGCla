import torch
import pandas as pd
from torch.utils.data import Dataset


class OriSet(Dataset):
    def __init__(self, filePath):
        print("file loading")
        self.df = pd.read_csv(filePath, header=None)  # 取出除第一行外的数据列
        self.data = self.df.iloc[:, 1:].values
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
        print("file loading finished")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, id):
        data = torch.tensor(self.data[id]).view(1, -1).float()
        label = torch.tensor(self.label[id]).long()
        return data, label


class PreSet(Dataset):
    def __init__(self, filePath, maskFrom, maskTo):
        print("file loading")
        self.data = pd.read_csv(filePath, header=None).iloc[:, 1:].values  # 取出除第一行外的数据列
        self.maskData = self.data.copy()
        for id in range(len(self.maskData)):
            self.maskData[id, maskFrom:maskTo] = 0
        print("file loading finished")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        maskData = torch.tensor(self.maskData[id]).view(1, -1).float()
        data = torch.tensor(self.data[id]).view(1, -1).float()
        if maskData.equal(data):
            print("Oh no")
        return maskData, data


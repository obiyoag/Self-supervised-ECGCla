import wfdb
import pandas as pd
import numpy as np


def delNonBeat(symbol, sample):  # 删除非心拍位置的标注
    mit_beat_codes = ['N', 'f', 'e', '/', 'j', 'n', 'B', 'L', 'R', 'S',
                      'A', 'J', 'a', 'V', 'E', 'r', 'F', 'Q', '?']
    symbol = np.array(symbol)  # symbol对应标签,sample为R峰所在位置，sig为R峰值
    isin = np.isin(symbol, mit_beat_codes)
    symbol = symbol[isin]  # 去除19类之外的非心拍标注信号
    return symbol, np.copy(sample[isin])


# 读取信号
path = "ECGCla/data/MIT-BIH/232"
record = wfdb.rdrecord(path, sampfrom=0, sampto=650000, channels=[1])
record = record.p_signal.reshape((650000,))

# 读取标记
ann = wfdb.rdann(path, 'atr', sampfrom=0, sampto=650000)
rSymboList, rSpotList = delNonBeat(ann.symbol, ann.sample)

#  准备截取
length = len(rSpotList)
recordSegList = []

#  开始截取
for pos in range(length):
    if rSymboList[pos] == 'N':
        rSpot = rSpotList[pos]
        recordSeg = record[rSpot - 100:rSpot + 150]  # 找到R波，前溯100个采样点，后取150个采样点，约0.7秒为一个心拍
        recordSegList.append(recordSeg)
        
# 讲截取的心拍变为Dataframe格式并保存为csv文件
df = pd.DataFrame(recordSegList)
df.to_csv('ECGCla/data/nSeg.csv')

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


def segment(record, rSpotList, rAnnList, segTypeList, type):
    #  准备截取
    length = len(rSpotList)
    index = 0
    sampleNum = 0
    #  统计需要截取的样本数
    for pos in range(length):
        if rAnnList[pos] in segTypeList:
            rSpot = rSpotList[pos]
            recordSeg = record[rSpot - 100:rSpot + 150]  # 找到R波，前溯100个采样点，后取150个采样点，约0.7秒为一个心拍
            if len(recordSeg) == 250:  # recordSeg数组长度不为250则不统计
                sampleNum += 1

    df = pd.DataFrame(np.zeros((sampleNum, 251)))  # 创建Dataframe数据类型用于保存

    #  开始截取
    for pos in range(length):
        if rAnnList[pos] in segTypeList:
            rSpot = rSpotList[pos]
            recordSeg = record[rSpot - 100:rSpot + 150]  # 找到R波，前溯100个采样点，后取150个采样点，约0.7秒为一个心拍
            if len(recordSeg) == 250:  # recordSeg数组长度不为250则不保存
                recordSeg = np.array(recordSeg)
                df.iloc[index, 0] = type
                df.iloc[index, 1:] = recordSeg
                index += 1

    df.to_csv('ECGCla/data/classified' + type + 'Seg.csv', mode='a', index=False, header=False)


def classify(type, typeList):
    # 读取信号
    for recordIndex in recordList:
        path = "ECGCla/data/MIT-BIH/" + recordIndex
        record = wfdb.rdrecord(path, sampfrom=0, sampto=650000, channels=[1])
        record = record.p_signal.reshape((650000,))

        # 读取标记
        ann = wfdb.rdann(path, 'atr', sampfrom=0, sampto=650000)
        rAnnList, rSpotList = delNonBeat(ann.symbol, ann.sample)

        segment(record, rSpotList, rAnnList, typeList, type)


if __name__ == "__main__":
    # 定义AAMI分类列表和信号列表
    nList = ['N', 'L', 'R']
    sList = ['A', 'a', 'S', 'J' 'e', 'j']
    vList = ['V', 'E']
    fList = ['F']
    qList = ['/', 'f', 'Q']
    recordList = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122', '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230',
                  '100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234']

    classify('F', fList)

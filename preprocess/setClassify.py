import pandas as pd


def getDataSeg(type, fromP, toP):

    index = range(fromP, toP)
    path = 'ECGCla/data/Classified/' + type + 'Seg.csv'
    df = pd.read_csv(path, header=None)
    dataSeg = df.iloc[index, :]
    return dataSeg


def saveDF(df, saveFileName):
    df.to_csv('ECGCla/data/Classified/' + saveFileName, index=False, header=False)


if __name__ == "__main__":

    NSeg = getDataSeg('N', 2400, 3000)
    SSeg = getDataSeg('S', 2400, 2927)
    VSeg = getDataSeg('V', 2400, 3000)
    FSeg = getDataSeg('F', 760, 802)

    TotalSeg = pd.concat([NSeg, SSeg, VSeg, FSeg])

    print("Item num in the generated dataset is {}.".format(len(TotalSeg)))

    saveDF(TotalSeg, 'testSet.csv')

import matplotlib.pyplot as plt
from dataSets import PreSet, OriSet
from models import UNet0, UNet1, UNet2
import torch


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preset1 = OriSet('ECGCla/data/Classified/preSet.csv', snr=-6)
    preset2 = OriSet('ECGCla/data/Classified/preSet.csv', snr=-100)

    data1, label1 = preset1[0]
    data2, label2 = preset2[0]

    # maskData1 = torch.unsqueeze(maskData01, 0)
    # maskData2 = torch.unsqueeze(maskData02, 0)

    # model = UNet2().to(device)
    # model.load_state_dict(torch.load('ECGCla/models2save/snr=-6.pkl'))

    # reData = maskData.to(device)
    # reData = model(maskData).view(-1).detach().numpy()

    index = range(250)

    plt.figure(1)

    plt.subplot(211)
    plt.plot(index, data1.view(-1))

    plt.subplot(212)
    # plt.plot(index, maskData0.view(-1))
    plt.plot(index, data2.view(-1))

    # plt.subplot(313)
    # plt.plot(index, reData)

    plt.show()

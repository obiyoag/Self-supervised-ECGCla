import matplotlib.pyplot as plt
from dataSets import PreSet
from models import UNet
import torch


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preset = PreSet('ECGCla/data/Classified/preSet.csv', 100, 150)
    maskData, data = preset[1]
    maskData = torch.unsqueeze(maskData, 0)

    model = UNet().to(device)
    model.load_state_dict(torch.load('ECGCla/models2save/premodel.pkl'))

    reData = maskData.to(device)
    reData = model(maskData).view(-1).detach().numpy()

    index = range(reData.shape[0])

    plt.figure(1)
    plt.subplot(211)
    plt.plot(index, reData)

    plt.subplot(212)
    plt.plot(index, data.view(-1))
    plt.show()

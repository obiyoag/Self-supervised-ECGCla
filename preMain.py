import torch
from models import UNet
from dataSets import PreSet
from torchsummary import summary
from torch.optim import Adam
from utils import getGenerators, pre_train, pre_validation

batch_size = 32
validation_split = .5
split_random_seed = 40
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
preset = PreSet('ECGCla/data/Classified/preSet.csv', 100, 150)

train_loader, val_loader = getGenerators(preset, batch_size, validation_split, split_random_seed)

model = UNet().to(device)
optimizer = Adam(model.parameters(), lr=1e-4)

summary(model, (1, 250))

for epoch in range(1, epochs + 1):
    pre_train(model, device, train_loader, optimizer, epoch)
    pre_validation(model, device, val_loader)
torch.save(model.state_dict(), 'ECGCla/models2save/premodel.pkl')

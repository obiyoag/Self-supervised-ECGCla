import torch
from dataSets import OriSet
from models import CNN
from torchsummary import summary
from torch.optim import Adam
from utils import tune_train, tune_test, getGenerators


batch_size = 32
test_split = .99
split_random_seed = 40
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

oriset = OriSet('ECGCla/data/Classified/trainSet.csv')
train_loader, test_loader = getGenerators(oriset, batch_size, test_split, split_random_seed)

CNN = CNN().to(device)
# model_dict = CNN.state_dict()
# pretrained_dict = torch.load('ECGCla/models2save/premodel.pkl')
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# CNN.load_state_dict(model_dict)

optimizer = Adam(CNN.parameters(), lr=1e-4)

summary(CNN, (1, 250))

for epoch in range(1, epochs + 1):
    tune_train(CNN, device, train_loader, optimizer, epoch)
    tune_test(CNN, device, test_loader)

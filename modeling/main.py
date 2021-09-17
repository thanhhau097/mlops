import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from modeling.dataset import MNISTDataset
from modeling.model import Model
from modeling.train import Trainer


def train_model():
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = MNISTDataset(root_folder='./data/mnist/train', transform=transform)
    val_dataset = MNISTDataset(root_folder='./data/mnist/test', transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model().to(device)
    loss_fn = F.nll_loss
    optimizer = torch.optim.Adam(model.parameters())
    num_epochs = 100

    trainer = Trainer(model, loss_fn, optimizer, device)
    trainer.train(num_epochs, train_dataloader, val_dataloader)


if __name__ == '__main__':
    train_model()
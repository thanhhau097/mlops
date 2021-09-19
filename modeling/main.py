import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import numpy as np

from modeling.dataset import MNISTDataset
from modeling.model import Model
from modeling.train import Trainer


def train_model(config):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = MNISTDataset(root_folder='/home/lionel/Desktop/MLE/mlops/data/mnist/train', transform=transform)
    val_dataset = MNISTDataset(root_folder='/home/lionel/Desktop/MLE/mlops/data/mnist/test', transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model().to(device)
    loss_fn = F.nll_loss
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
    num_epochs = 10

    trainer = Trainer(model, loss_fn, optimizer, device)
    best_model = trainer.train(num_epochs, train_dataloader, val_dataloader)
    torch.save(best_model.state_dict(), './weights/mnist_model.pt')


def hyperparams_opt():
    search_space = {
        "lr": tune.sample_from(lambda spec: 10**(-10 * np.random.rand())),
        "momentum": tune.uniform(0.1, 0.9)
    }
    analysis = tune.run(
        train_model,
        num_samples=4,
        scheduler=ASHAScheduler(metric="mean_accuracy", mode="max"),
        config=search_space
    )
    # # Obtain a trial dataframe from all run trials of this `tune.run` call.
    # dfs = analysis.trial_dataframes

    # # Plot by epoch
    # ax = None  # This plots everything on the same plot
    # for d in dfs.values():
    #     ax = d.mean_accuracy.plot(ax=ax, legend=False)

if __name__ == '__main__':
    train_model(config={
        "lr": 0.001,
        "momentum": 0.9
    })
    # hyperparams_opt()
    
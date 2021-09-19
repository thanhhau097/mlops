import torch
from torch.nn import functional as F


class Trainer():
    def __init__(
        self, 
        model,
        loss_fn=None,
        optimizer=None,
        device=torch.device("cpu"),
    ):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train_step(self, dataloader):
        self.model.train()
        loss = 0.0
        for _, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            batch_loss = self.loss_fn(output, target)
            batch_loss.backward()
            self.optimizer.step()

            loss += batch_loss.detach().item()

        return loss / len(dataloader)

    def eval_step(self, dataloader):
        self.model.eval()
        loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                batch_loss = self.loss_fn(output, target, reduction='mean')
                loss += batch_loss.detach().item()
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        return loss / len(dataloader), correct / len(dataloader.dataset)

    def train(
        self, 
        num_epochs,
        train_dataloader,
        eval_dataloader,
    ):
        best_loss = 1e10
        best_model = None
        for epoch in range(num_epochs):
            loss = self.train_step(train_dataloader)
            eval_loss, eval_acc = self.eval_step(eval_dataloader)

            if eval_loss < best_loss:
                best_loss = eval_loss
                best_model = self.model

            print('Epoch: {}/{} | Loss: {} | Eval Loss: {} | Eval Acc: {}'.format(
                epoch + 1,
                num_epochs,
                loss,
                eval_loss,
                eval_acc,
            ))
        return best_model

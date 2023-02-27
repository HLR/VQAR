import torch
import math

class Trainer:

    def __init__(self, lr):

        self.model = torch.nn.Sequential(
            torch.nn.Linear(3, 1),
            torch.nn.Flatten(0, 1)
            )
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr)

    def forward(self, x, y):
        y_pred = self.model(x)
        new_y_pred = torch.zeros(3, requires_grad=False)
        new_y_pred[0] = y_pred
        new_y = torch.zeros(3, requires_grad=False)
        new_y[0] = y
        loss = self.loss_fn(new_y_pred, new_y)
        return loss

    def train(self, xs, ys):
        self.model.train()
        self.optimizer.zero_grad()
        for x, y in zip(xs, ys):
            loss = self.forward(x.reshape(1, -1), y.reshape(1))
            loss.backward()
        self.optimizer.step()
        return loss.item()


if __name__ == "__main__":
    x = torch.linspace(-math.pi, math.pi, 2000)
    y = torch.sin(x)
    p = torch.tensor([1, 2, 3])
    xx = x.unsqueeze(-1).pow(p)
    lr = 1e-3

    trainer = Trainer(lr)
    loss = trainer.train(xx, y)

    print("end")
import os
import torch
from matplotlib.axes import Axes
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm

from programs import Model

BATCH_SIZE = 512
EPOCHS = 500
LR = 0.1
ROWS = 3

CTR = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class State:
    def __init__(self):
        self.model = None
        self.epoch = 0
        self.batch = 0
        self.steps = 0
        self.writer = SummaryWriter()

    @staticmethod
    def log(s):
        # sample log
        return s % int(np.sqrt(s) + 1) == 0

    def add_scalar(self, tag, scalar_value):
        if not self.log(self.steps):
            return
        try:
            self.writer.add_scalar(tag=tag, scalar_value=scalar_value, global_step=self.steps)
        except ValueError:
            pass

    def add_histogram(self, tag, values):
        if not self.log(self.steps):
            return
        try:
            self.writer.add_histogram(tag=tag, values=values, global_step=self.steps)
        except ValueError:
            pass


def visualize(model):
    indices = torch.arange(0, ROWS * ROWS).repeat(16)
    pos = conv_to_pos(indices)
    x = pos[0].cpu().detach().numpy()
    (y1, w), y2, mask = model(*pos)
    y = y1.view(16, ROWS * ROWS, model.num_units)
    y = y.permute(2, 0, 1).view(model.num_units, -1)
    y = y.cpu().detach().numpy()

    fig, a = plt.subplots(ROWS, ROWS, figsize=(6, 6))
    for i, val in enumerate(y):
        axis: Axes = a[i//ROWS][i%ROWS]
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
        axis.scatter(x=x[:,0], y=x[:,1], c=val)
    plt.show()
    # return imgs


def prepare_data_loader():
    places = ROWS * ROWS

    data = np.random.randint(0, places, size=BATCH_SIZE * 10)

    data_loader = DataLoader(dataset=data,
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             drop_last=True)
    return data_loader


def adjust_learning_rate(optimizers):
    """Sets the learning rate to the initial LR decayed by 10 every 180 epochs"""
    epoch = state.epoch
    lr = LR * (0.1 ** (epoch // 180))
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def cal_loss(y1, y2, w, mask, t):
    """
    :param y1:  s_b * n_u
    :param y2:  s_b * n_u
    :param w:   s_b(q) * s_b * num_units
    :param mask: s_b * num_units
    :param t:   n_u
    """

    def cal_norm(x):
        return torch.pow(torch.relu(-x), 2).mean()

    l_1_2 = torch.exp(-torch.abs(y1 - y2).clamp_max(5))  # s_b * n_u
    l_1_neg = torch.sum(w * torch.exp(-torch.abs(y1.unsqueeze(1) - y1).clamp_max(5)), dim=1)  # s_b * n_u
    l_2_neg = torch.sum(w * torch.exp(-torch.abs(y2.unsqueeze(1) - y1).clamp_max(5)), dim=1)

    loss = -torch.log(l_1_2 / l_1_neg) - torch.log(l_1_2 / l_2_neg)
    loss_all = loss.mean()
    loss = loss.masked_fill(~mask, 0)
    loss = loss.mean() + cal_norm(y1)
    # loss = loss.mean()

    state.add_histogram(tag='y1', values=y1)
    state.add_histogram(tag='l12', values=l_1_2)
    state.add_histogram(tag='l1neg', values=l_1_neg)
    state.add_histogram(tag='l2neg', values=l_2_neg)
    state.add_scalar(tag='loss', scalar_value=loss_all.item())

    return loss


def flip_grad(optimizer):
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            p.grad.data = - p.grad.data


def optimize(loss, optimizers):
    for optim in optimizers:
        optim.zero_grad()
    loss.backward()
    optim = optimizers[0]
    optim.step()
    optim = optimizers[1]
    flip_grad(optim)
    optim.step()


def conv_to_pos(indices):
    row = indices // ROWS
    vol = indices % ROWS
    pos = torch.stack([row, vol], dim=1).float().to(device)

    noise1 = 0.2 * torch.randn_like(pos)
    pos1 = pos + noise1
    pos1 = pos1 / ROWS - 0.5

    noise2 = 0.2 * torch.randn_like(pos)
    pos2 = pos + noise2
    pos2 = pos2 / ROWS - 0.5

    return pos1, pos2


def train_epoch(model, data_loader, optimizers, ):
    for batch in data_loader:
        batch = conv_to_pos(batch)
        (y1, w), y2, mask = model(*batch)
        loss = cal_loss(y1, y2, w, mask, model.temperature)
        optimize(loss, optimizers)
        state.steps += 1
    pass


def train(model, data_loader, optimizers):
    # visualize(model, )
    for epoch in tqdm(range(EPOCHS)):
        state.epoch = epoch
        adjust_learning_rate(optimizers)
        train_epoch(model, data_loader, optimizers, )
        # visualize(model, )
        torch.save(model.state_dict(), f=os.path.join('state_dicts', 'model.state_dict.' + str(epoch)))


def main():
    global state
    state = State()
    model: Model = Model()
    sd = torch.load(f='state_dicts/model.state_dict.199')
    model.load_state_dict(sd)
    model = model.to(device)
    state.model = model
    data_loader = prepare_data_loader()
    optimizers = [torch.optim.SGD(params=(*model.encoder.parameters(), model.temperature),
                                  lr=LR, momentum=0.9, weight_decay=0.001),
                  torch.optim.SGD(params=model.context_net.parameters(),
                                  lr=LR, momentum=0.9, weight_decay=0.001), ]

    train(model, data_loader, optimizers)
    visualize(model)


if __name__ == '__main__':
    os.chdir('..')
    main()

import torch
import torch.nn as nn

from .modules import Encoder
from .modules import ContextNet

NUM_UNITS = 9
MASK_P = 0.2
SIZE = 2


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_units = NUM_UNITS
        self.size = SIZE
        self.encoder = Encoder(num_units=NUM_UNITS, dim_inputs=SIZE, dim_hidden=SIZE * 2)
        self.context_net = ContextNet(num_units=NUM_UNITS, dim_hidden=NUM_UNITS * 4, mask_p=MASK_P)
        self.temperature = nn.Parameter(torch.ones(NUM_UNITS))

    def forward(self, x1, x2):
        y1 = self.encoder(x1)  # s_b * n_u
        w, mask = self.context_net(y1)  # s_b(q) * s_b * num_units

        y2 = self.encoder(x2)

        return (y1, w), y2, mask

    def encode(self, x):
        x = x.view(-1, SIZE)
        y = self.encoder(x)

        return y

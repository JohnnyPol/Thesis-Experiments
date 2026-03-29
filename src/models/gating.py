
from torch import nn

class Gmodel(nn.Module):

  def __init__(self, hidden_dim):
        super(Gmodel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

  def forward(self, x):
        return self.net(x).squeeze(-1)
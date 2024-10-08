import torch 
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, layers, lr=1e-3):
        super(PINN, self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

        self.layers.apply(self.init_weights)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.timer = 0
        self.to(self.device)

    def init_weights(self, m):
      if isinstance(m, nn.Linear):
          torch.nn.init.xavier_uniform_(m.weight)
          torch.nn.init.normal_(m.bias)

    def forward(self, xy):

        self.timer += 1
        dist = ((xy[:,0:1] + 0.5) * (xy[:,0:1] - 0.5))

        for i in range(len(self.layers) - 1):
            xy = torch.tanh(self.layers[i](xy))
        xy = self.layers[-1](xy)

        # displacements
        phi, w, mxx_x, qx_x = xy.split(1, dim=1)
        w = w*dist
        phi = phi

        # Concatenate along the last dimension
        return torch.cat([phi, w, mxx_x, qx_x], dim=1)
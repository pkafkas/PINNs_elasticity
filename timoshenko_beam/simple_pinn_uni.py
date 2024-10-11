import torch 
import torch.nn as nn

class PINN(nn.Module):
    """ This class represents a Physics-Informed Neural Network (PINN) designed to address the Timoshenko beam bending problem.
    Attributes:

    layers: An nn.ModuleList containing fully connected layers that structure the neural network.
    criterion: Utilizes nn.MSELoss as the loss function.
    optimizer: Implements the Adam optimizer (torch.optim.Adam) for optimization during the training.
    device: Specifies the hardware for training, either 'cuda' for GPU or 'cpu'.

    Methods:

    forward: Defines how the neural network computes outputs from the input data during the forward pass. """

    def __init__(self, layers, lr=1e-3, dist=False):
        super(PINN, self).__init__()
        
        self.dist = dist
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

        for i in range(len(self.layers) - 1):
            xy = torch.tanh(self.layers[i](xy))
        xy = self.layers[-1](xy)

        # displacements
        q, w = xy.split(1, dim=1)

        # Concatenate along the last dimension
        return torch.cat([q, w], dim=1)
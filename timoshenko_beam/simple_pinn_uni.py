import torch 
import torch.nn as nn

class PINN(nn.Module):
    """
    This class defines a Physics-Informed Neural Network (PINN) for solving differential equations,
    specifically tailored for the Timoshenko beam bending problem.

    Attributes:
    -----------
    layers : nn.ModuleList
        A list of fully connected layers that define the architecture of the neural network.
    criterion : nn.MSELoss
        The mean squared error loss function used to train the network.
    optimizer : torch.optim.Adam
        Adam optimizer used for gradient-based optimization during training.
    device : str
        The device on which the model is trained, either 'cuda' (if GPU is available) or 'cpu'.

    Methods:
    --------
    forward:
        Defines the forward pass of the neural network, computing the output based on the input data.
    """

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
import torch
import matplotlib.pyplot as plt

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs = grad_outputs, create_graph=True)[0]
    return grad

def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def plot_loss(model, eq):
    with torch.no_grad():
        x = torch.linspace(-1.0, 1.0, steps=100) / 2
        x = x.unsqueeze(1)
        _, w, _, _ = model(x).split(1, dim=1)
        
        plt.plot(x, w)
        plt.xlabel("length (m)")
        plt.ylabel("displacement (m)")
        plt.show()


def getData(num_points, grad=False):
    x = torch.linspace(-0.9, 0.9, steps=num_points) / 2
    x = x.unsqueeze(1)
    return x.requires_grad_(grad)

def getBoundaryData(num_points, grad=False):
    # Ensure that N is even
    assert num_points % 2 == 0, "N must be an even number."
    
    # Create a tensor with N/2 zeros and N/2 ones
    half_N = num_points // 2
    sampled_points = torch.cat((torch.zeros(half_N), torch.ones(half_N)))
    
    # Shuffle the tensor randomly
    sampled_points = sampled_points[torch.randperm(num_points)] - 0.5
    sampled_points = sampled_points.unsqueeze(1)
    
    return sampled_points.requires_grad_(grad)


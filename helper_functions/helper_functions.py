import torch

def gradient(y, x, grad_outputs=None):
    """Compute dy/dx @ grad_outputs"""
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs = grad_outputs, create_graph=True)[0]
    return grad

def jacobian(y, x):
    """Compute dy/dx = dy/dx @ grad_outputs; 
    for grad_outputs in [1, 0, ..., 0], [0, 1, 0, ..., 0], ...., [0, ..., 0, 1]"""
    jac = torch.zeros(y.shape[0], x.shape[0]) 
    for i in range(y.shape[0]):
        grad_outputs = torch.zeros_like(y)
        grad_outputs[i] = 1
        jac[i] = gradient(y, x, grad_outputs = grad_outputs)
    return jac

def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def plot_loss(model, L=1.0, U=1.0):
    with torch.no_grad():

        x = torch.linspace(-1.0, 1.0, steps=100) / 2
        x = x.unsqueeze(1)

        _, w, _, _ = model(x).split(1, dim=1)
        #fem =  0.00252
        #max = np.abs(solution_w).max()
        #rel_diff = (np.abs(max - fem) / fem)*100
        #print(f'max w: {max}, fem: {fem}, Relative Difference: {rel_diff}')
        plt.plot(x, w)
        #plt.xlabel("length (m)")
        #plt.ylabel("width (m)")
        #plt.colorbar().set_label('w (m)')
        #plt.colorbar()
        plt.show()


def getData(num_points, grad=False):
    x = torch.linspace(-1.0, 1.0, steps=num_points) / 2
    x = x.unsqueeze(1)
    return x.requires_grad_(grad)
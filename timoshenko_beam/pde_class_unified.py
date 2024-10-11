import torch
import torch.nn as nn

class Pde():
    """ This class specifies the parameters and physical properties necessary to compute the governing equations and derivatives for the 
    Timoshenko beam bending problem.
    
    Attributes:

    EI: A float representing the flexural rigidity of the beam, which is the product of the moment of inertia (I) and Young's modulus (E).
    kAG: A float that denotes the shear stiffness.
    q: A float for the applied distributed load acting on the beam.
    L: A float that represents the length of the beam. """

    def __init__(self, E, nu=0.3, h=0.1, k=5/6, l=1.0, q=1.0):
        I = (h**3)/12
        G = E/(2*(1+nu))
        A = h
        self.EI = I * E
        self.D = (E * h**3) / (12*(1-nu**2))
        self.kAG = k * A * G
        self.q = q
        self.L = l

        self.criterion = nn.MSELoss()

    
    def get_w(self, x):
        P = self.q
        L = self.L
        D = self.D
        S = self.kAG

        # First term: P * L^4 / 24D * (x / L - 2 * x^3 / L^3 + x^4 / L^4)
        term1 = (P * L**4) / (24 * D) * (x / L - 2 * (x**3) / (L**3) + (x**4) / (L**4))

        # Second term: P * L^2 / 2S * (x / L - x^3 / L^3)
        term2 = (P * L**2) / (2 * S) * (x / L - (x**3) / (L**3))

        # Return the sum of the two terms
        return term1 + term2


    def getLoss(self, model, x):
        q, w = model(x).split(1, dim=1)

        # Compute derivatives
        w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(w_x), create_graph=True)[0]
        w_xxx = torch.autograd.grad(w_xx, x, grad_outputs=torch.ones_like(w_xx), create_graph=True)[0]
        w_xxxx = torch.autograd.grad(w_xxx, x, grad_outputs=torch.ones_like(w_xxx), create_graph=True)[0]

        q_x = torch.autograd.grad(q, x, grad_outputs=torch.ones_like(q), create_graph=True)[0]
        q_xx = torch.autograd.grad(q_x, x, grad_outputs=torch.ones_like(q_x), create_graph=True)[0]
    
        eq = self.EI * w_xxxx - self.q + (self.EI/self.kAG)*q_xx


        resloss = self.criterion(eq, torch.zeros_like(eq))

        return resloss
import torch
import torch.nn as nn
from helper_functions import gradient, laplace

class Pde():
    """
    This class defines the parameters and physical properties needed to compute the governing 
    equations and derivatives for the Timoshenko beam bending problem. It encapsulates the 
    material properties (Young's modulus, Poisson's ratio), geometric properties (beam height, 
    length), and loading conditions (applied load, shear correction factor) required to compute 
    the bending and shear deformations according to Timoshenko beam theory.

    Attributes:
    -----------
    IE : float
        Flexural rigidity of the beam, calculated as the product of the moment of inertia (I) 
        and Young's modulus (E).
    D : float
        Flexural rigidity factor for the plane stress condition, adjusted for Poisson's ratio (nu).
    kAG : float
        Shear stiffness, calculated using the shear correction factor (k), cross-sectional area (A),
        and shear modulus (G).
    q : float
        Applied distributed load on the beam.
    L : float
        Length of the beam.
    """

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

    def getDerivs(self, w, phi, x):
        dphi_dx = gradient(phi, x)
        return dphi_dx
    
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
        q, w, mxx, qx = model(x).split(1, dim=1)

        #phi_x_t = gradient(phi, x)
        w_xx = laplace(w, x)
        w_xxxx = laplace(w_xx, x)
        q_xx = laplace(q ,x)

        #mxx_target = -self.IE * gradient(phi, x)
        #qx_target = self.kAG * (-phi + gradient(w, x))
    
        eq = self.EI * w_xxxx - self.q + (self.EI/self.kAG)*q_xx
        
        #mxx_x = gradient(mxx,x)
        #qx_x= gradient(qx, x)


        resloss = self.criterion(eq, torch.zeros_like(eq))

        return resloss
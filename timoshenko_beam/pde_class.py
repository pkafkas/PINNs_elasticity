import torch
import torch.nn as nn
from helper_functions.helper_functions import gradient

class Pde():
    def __init__(self, E, nu=0.3, h=0.1, k=5/6):
        I = (h**3)/12
        print(I)
        G = E/(2*(1+nu))
        A = h
        self.IE = I * E
        self.kAG = k * A * G
        self.criterion = nn.MSELoss()

    def getDerivs(self, w, phi, x):
        dphi_dx = gradient(phi, x)
        return dphi_dx


    def getLoss(self, model, x):
        #xz = torch.cat([x, z], dim=1)
        #w, phix, phiy, w_x, w_y, phix_x, phiy_y, phix_y, phiy_x = model(x,y).split(1, dim=1)
        phi, w, mxx, qx = model(x).split(1, dim=1)

        phi_x_t = gradient(phi, x)
        w_x_t = gradient(w, x)
        mxx_target = -self.IE * gradient(phi, x)
        qx_target = self.kAG * (-phi + gradient(w, x))
        q = -1.0#/len(x)



        derivloss = self.criterion(mxx, mxx_target) +\
                      self.criterion(qx, qx_target)


        mxx_x = gradient(mxx,x)
        qx_x= gradient(qx, x)

        eq1 = mxx_x - qx
        eq2 = qx_x + q

        res1 = self.criterion(eq1, torch.zeros_like(eq1))
        res2 = self.criterion(eq2, torch.zeros_like(eq2))

        resloss = res1 + res2

        return resloss, derivloss
from AIScratch.NeuralNetwork.Optimizers import Optimizer
import numpy as np

class ADAMOptimizer(Optimizer):
    def __init__(self, n_p1, n_p, eta, epsilon, beta1, beta2):
        super().__init__()
        self.eta = eta
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta1t = beta1 
        self.beta2t = beta2
        self.mp = np.zeros((n_p, n_p1))
        self.vp = np.zeros((n_p, n_p1))
        self.mb = np.zeros(n_p)  # m biais
        self.vb = np.zeros(n_p)  # v biais

    def update_mp(self, grad_L_w):
        self.mp = self.beta1 * self.mp + (1 - self.beta1) * grad_L_w
    def update_vp(self, grad_L_w):
        self.vp = self.beta2 * self.vp + (1 - self.beta2) * (grad_L_w ** 2) 
    def update_mb(self, grad_L_zp):
        self.mb = self.beta1 * self.mb + (1 - self.beta1) * grad_L_zp
    def update_vb(self, grad_L_zp):
        self.vb = self.beta2 * self.vb + (1 - self.beta2) * (grad_L_zp ** 2) 

    def optimize(self, errors, gradients, inputs):
        grad_L_z = errors * gradients # vector 
        self.update_mb(grad_L_z)
        self.update_vb(grad_L_z)
        grad_L_w = np.outer(grad_L_z, inputs) # matrix
        self.update_mp(grad_L_w)
        self.update_vp(grad_L_w)
        mp_corr = self.mp / (1 - self.beta1t)
        vp_corr = self.vp / (1 - self.beta2t)
        mb_corr = self.mb / (1 - self.beta1t)
        vb_corr = self.vb / (1 - self.beta2t)
        update_w = self.eta / (np.sqrt(vp_corr) + self.epsilon)
        update_b = self.eta / (np.sqrt(vb_corr) + self.epsilon) * mb_corr
        self.beta1t *= self.beta1
        self.beta2t *= self.beta2
        return update_w, mp_corr, update_b
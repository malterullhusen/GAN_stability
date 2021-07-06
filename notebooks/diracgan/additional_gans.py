"""Additional GANs for the Dirac-GAN [1] experiment.

# References
    
[1] Mescheder, Lars, Andreas Geiger, and Sebastian Nowozin. "Which 
    training methods for GANs do actually converge?." In International 
    conference on machine learning, pp. 3481-3490. PMLR, 2018.
"""

from diracgan.gans import fp
from diracgan.gans import VectorField
import numpy as np


class AdditionalGAN(VectorField):
    """Reference implementation of an additional 
    GAN for the Dirac-GAN experiment.
    """

    def __init__(self, arg):
        super().__init__()
        self.arg = arg

    def _get_vector(self, theta, psi):
        v1 = 0
        v2 = 0
        # additional code here
        return v1, v2

    def _postprocess(self, theta, psi):
        # additional code here
        return theta, psi

class RegularizedGAN(VectorField):
    """A GAN that takes the gradient of a regularization 
    function as a constructor argument. The regularization 
    function represents the gradient of a regularization 
    term with respect to psi. It should be a function of 
    two floats or two numpy arrays representing theta and 
    psi and return a float or a numpy array of the same 
    shape as the input arrays. 
    
    For the real-valued function f in the loss term the 
    standard function f(x) = -log(1 + exp(-x)) is used.
    """

    def __init__(self, grad_R, reg_lambda = 0.1):
        super().__init__()
        self.grad_R = grad_R
        self.reg_lambda = reg_lambda

    def _get_vector(self, theta, psi):
        v1 = -psi * fp(psi*theta)
        v2 = theta * fp(psi*theta)
        v2 -= self.reg_lambda * self.grad_R(theta, psi)
        return v1, v2

class RegLimitedDataGAN(VectorField):
    """A regularized GAN for limited data [1].

    In normal mode, i.e. when simulating a trajectory, the 
    regularization term depends on the past outputs of the 
    discriminator function. When this class is used to 
    generate a vector field plot, at each point all past 
    discriminator outputs are set to zero. Therefore a vector 
    field plot can not fully capture the behaviour of this 
    GAN.

    # References

    [1] Tseng, Hung-Yu, Lu Jiang, Ce Liu, Ming-Hsuan Yang, and 
        Weilong Yang. "Regularizing Generative Adversarial Networks 
        under Limited Data." In Proceedings of the IEEE/CVF 
        Conference on Computer Vision and Pattern Recognition, 
        pp. 7921-7931. 2021.
    """

    def __init__(self, reg_lambda = 0.1):
        super().__init__()
        self.reg_lambda = reg_lambda
        # gamma value used in original paper
        self.gamma = 0.99 
        # moving average of discriminator outputs for generator 
        # samples. The regularizer usually also contains a moving 
        # average of discriminator outputs for real samples, but 
        # for Dirac-GAN this average is always zero.
        self.a_F_prev = 0 

    def _a_current(self, a_prev, v_current):
        return self.gamma * a_prev + (1 - self.gamma) * v_current

    def _grad_R_LC(self, a_F, D_F, theta):
        return 2 * a_F * self._grad_a_F(theta) + 2 * D_F * theta

    def _grad_a_F(self, theta):
        return (1 - self.gamma) * theta

    def _get_vector(self, theta, psi):
        D_F = psi * theta # discriminator output for generator sample
        v1 = -psi * fp(D_F) # standard GAN loss
        v2 = theta * fp(D_F) # standard GAN loss
        a_F = None
        # if theta is a float, we assume that this function is used 
        # to simulate a trajectory and update self.a_F_prev. Otherwise 
        # we assume that this function is used to generate a vector 
        # field plot and do not use self.a_F_prev 
        if isinstance(theta, float):
            a_F = self._a_current(self.a_F_prev, D_F)
            self.a_F_prev = a_F
        else:
            a_F = self._a_current(0, D_F)
        # applying the regularization
        v2 -= self.reg_lambda * self._grad_R_LC(a_F, D_F, theta)
        return v1, v2

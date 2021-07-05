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
    """A GAN that takes a regularization function as a 
    constructor argument. The regularization function 
    represents the gradient of a regularization term with 
    respect to psi. It should be a function of two numpy 
    arrays representing theta and psi and return a numpy 
    array of the same shape as the input arrays. 
    
    For the real-valued function f in the loss term the 
    standard function f(x) = -log(1 + exp(-x)) is used.
    """

    def __init__(self, R):
        super().__init__()
        self.R = R

    def _get_vector(self, theta, psi):
        v1 = -psi * fp(psi*theta)
        v2 = theta * fp(psi*theta) - self.R(theta, psi)
        # additional code here
        return v1, v2

    def _postprocess(self, theta, psi):
        # additional code here
        return theta, psi
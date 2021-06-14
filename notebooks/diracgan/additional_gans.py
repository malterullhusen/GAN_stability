import numpy as np
from diracgan.gans import VectorField

class AdditionalGAN(VectorField):
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


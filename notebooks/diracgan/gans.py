import numpy as np
from diracgan.util import sigmoid, clip


class VectorField(object):
    """Base class for implementations of different GAN 
    regularization strategies in the context of Dirac-GAN. 
    
    Subclasses of this class are used in the Dirac-GAN experiment 
    to train an extremely simple GAN architecture with one 
    generator parameter theta and one discriminator parameter psi.

    Subclasses of this class must implement the [_get_vector] 
    function and may implement the [_postprocess] function.

    The [_get_vector] function takes two numpy arrays representing 
    the current GAN parameters theta and psi as arguments. The 
    function should return two numpy arrays representing their 
    gradients, called v1 and v2 here. During training, the 
    gradients will be multiplied with a learning rate parameter 
    and then added to the current values of theta and psi. 

    The [_postprocess] function is called after the above steps. It 
    takes the updated values of theta and psi as arguments and 
    returns postprocessed values for theta and psi. By default, this 
    function returns its arguments unchanged.

    Note, that the input arrays may contain more than one entry. In 
    this case the output shall be computed for all theta-psi pairs 
    and the output arrays shall have the same shape as the input 
    arrays.

    # Dirac-GAN

    The Dirac-GAN [1] is a simple GAN with one generator parameter theta 
    and one discriminator parameter psi. The generator distribution is 
    a delta peak at theta and the discriminator is given by the linear 
    function D(x) = psi * x. The true data distribution for this 
    experiment is a Dirac-distribution concentrated at 0.

    The loss function L(theta, psi) is defined as

        L(theta, psi) = f(psi * theta) + f(0)

    for a real-valued function f. 

    # References
    
    [1] Mescheder, Lars, Andreas Geiger, and Sebastian Nowozin. "Which 
        training methods for GANs do actually converge?." In International 
        conference on machine learning, pp. 3481-3490. PMLR, 2018.
    """

    def __call__(self, theta, psi):
        theta_isfloat = isinstance(theta, float)
        psi_isfloat = isinstance(psi, float)
        if theta_isfloat:
            theta = np.array([theta])
        if psi_isfloat:
            psi = np.array([psi])

        v1, v2 = self._get_vector(theta, psi)

        if theta_isfloat:
            v1 = v1[0]
        if psi_isfloat:
            v2 = v2[0]

        return v1, v2

    def postprocess(self, theta, psi):
        theta_isfloat = isinstance(theta, float)
        psi_isfloat = isinstance(psi, float)
        if theta_isfloat:
            theta = np.array([theta])
        if psi_isfloat:
            psi = np.array([psi])
        theta, psi = self._postprocess(theta, psi)
        if theta_isfloat:
            theta = theta[0]
        if psi_isfloat:
            psi = psi[0]

        return theta, psi

    def step_sizes(self, h):
        return h, h

    def _get_vector(self, theta, psi):
        raise NotImplemented

    def _postprocess(self, theta, psi):
        return theta, psi


# GANs
def fp(x):
    return sigmoid(-x)


def fp2(x):
    return -sigmoid(-x) * sigmoid(x)


class GAN(VectorField):
    """Standard GAN as introduced in: 
    
    Goodfellow, Ian, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, 
    David Warde-Farley, Sherjil Ozair, Aaron Courville, and 
    Yoshua Bengio. "Generative adversarial nets." Advances in 
    neural information processing systems 27 (2014).
    """
    def _get_vector(self, theta, psi):
        v1 = -psi * fp(psi*theta)
        v2 = theta * fp(psi*theta)
        return v1, v2


class NSGAN(VectorField):
    def _get_vector(self, theta, psi):
        v1 = -psi * fp(-psi*theta)
        v2 = theta * fp(psi*theta)
        return v1, v2


class WGAN(VectorField):
    def __init__(self, clip=0.3):
        super().__init__()
        self.clip = clip

    def _get_vector(self, theta, psi):
        v1 = -psi
        v2 = theta

        return v1, v2

    def _postprocess(self, theta, psi):
        psi = clip(psi, self.clip)
        return theta, psi


class WGAN_GP(VectorField):
    def __init__(self, reg=1., target=0.3):
        super().__init__()
        self.reg = reg
        self.target = target

    def _get_vector(self, theta, psi):
        v1 = -psi
        v2 = theta - self.reg * (np.abs(psi) - self.target) * np.sign(psi)
        return v1, v2


class GAN_InstNoise(VectorField):
    def __init__(self, std=1):
        self.std = std

    def _get_vector(self, theta, psi):
        theta_eps = (
            theta + self.std*np.random.randn(*([1000] + list(theta.shape)))
        )
        x_eps = (
            self.std * np.random.randn(*([1000] + list(theta.shape)))
        )
        v1 = -psi * fp(psi*theta_eps)
        v2 = theta_eps * fp(psi*theta_eps) - x_eps * fp(-x_eps * psi)
        v1 = v1.mean(axis=0)
        v2 = v2.mean(axis=0)
        return v1, v2


class GAN_GradPenalty(VectorField):
    def __init__(self, reg=0.3):
        self.reg = reg

    def _get_vector(self, theta, psi):
        v1 = -psi * fp(psi*theta)
        v2 = +theta * fp(psi*theta) - self.reg * psi
        return v1, v2


class NSGAN_GradPenalty(VectorField):
    def __init__(self, reg=0.3):
        self.reg = reg

    def _get_vector(self, theta, psi):
        v1 = -psi * fp(-psi*theta)
        v2 = theta * fp(psi*theta) - self.reg * psi
        return v1, v2


class GAN_Consensus(VectorField):
    def __init__(self, reg=0.3):
        self.reg = reg

    def _get_vector(self, theta, psi):
        v1 = -psi * fp(psi*theta)
        v2 = +theta * fp(psi*theta)

        # L  0.5*(psi**2 + theta**2)*f(psi*theta)**2
        v1reg = (
            theta * fp(psi*theta)**2
            + 0.5*psi * (psi**2 + theta**2) * fp(psi*theta)*fp2(psi*theta)
        )
        v2reg = (
            psi * fp(psi*theta)**2
            + 0.5*theta * (psi**2 + theta**2) * fp(psi*theta)*fp2(psi*theta)
        )
        v1 -= self.reg * v1reg
        v2 -= self.reg * v2reg

        return v1, v2


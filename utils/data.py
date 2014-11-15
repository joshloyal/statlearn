import numpy as np

def make_1Dregression(n_samples=100, noise=0.0, func=None, seed=None):
    """Generate a random 1d regression problem

    Parameters
    ----------
    n_samples : int, optional (default=100)
        the number of samples

    noise : float, optional (default=0.0)
        the standard deviation of the gaussian noise applied to the output

    func : lambda x : func(x) (default y = x)
        function to base the model (y = func(x)) on

    seed : int, optional
        seed for the random number generator

    Returns
    -------
    X : array shape [n_samples]
        the input samples

    y : array shape [n_samples]
        the output values
    """
    if seed is not None:
        np.random.seed(seed)
    X = np.random.randn(n_samples)

    if func is None:
        y = X
    else:
        y = func(X)

    if noise > 0.0:
        y = y + np.random.normal(scale=noise, size=y.shape)
    return X, y

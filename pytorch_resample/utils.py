import math
import random


__all__ = ['random_poisson']


def random_poisson(rate, rng=random):
    """Sample a random value from a Poisson distribution.

    This implementation is done in pure Python. Using PyTorch would be much slower.

    References:
        - https://www.wikiwand.com/en/Poisson_distribution#/Generating_Poisson-distributed_random_variables

    """

    L = math.exp(-rate)
    k = 0
    p = 1

    while p > L:
        k += 1
        p *= rng.random()

    return k - 1

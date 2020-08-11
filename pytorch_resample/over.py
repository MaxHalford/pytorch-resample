import random
import collections

import torch

from . import utils


class OverSampler(torch.utils.data.IterableDataset):
    """Dataset wrapper for over-sampling.

    Parameters:
        dataset
        desired_dist: The desired class distribution. The keys are the classes whilst the
            values are the desired class percentages. The values must sum up to 1.
        seed: Random seed for reproducibility.

    Attributes:
        actual_dist: The counts of the observed sample labels.
        rng: A random number generator instance.

    """

    def __init__(self, dataset: torch.utils.data.IterableDataset, desired_dist: dict,
                 seed: int = None):

        self.dataset = dataset
        self.desired_dist = {c: p / sum(desired_dist.values()) for c, p in desired_dist.items()}
        self.seed = seed

        self.actual_dist = collections.Counter()
        self.rng = random.Random(seed)
        self._pivot = None

    def __iter__(self):

        for x, y in self.dataset:

            self.actual_dist[y] += 1

            # To ease notation
            f = self.desired_dist
            g = self.actual_dist

            # Check if the pivot needs to be changed
            if y != self._pivot:
                self._pivot = max(g.keys(), key=lambda y: g[y] / f[y])
            else:
                yield x, y
                continue

            # Determine the sampling ratio if the observed label is not the pivot
            M = g[self._pivot] / f[self._pivot]
            rate = M * f[y] / g[y]

            for _ in range(utils.random_poisson(rate, rng=self.rng)):
                yield x, y

    @classmethod
    def expected_size(cls, n, desired_dist, actual_dist):
        M = max(
            actual_dist.get(k) / desired_dist.get(k)
            for k in set(desired_dist) | set(actual_dist)
        )
        return int(n * M)

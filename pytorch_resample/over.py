import collections
import random

import torch

from . import utils


class OverSampler(torch.utils.data.IterableDataset):
    """Dataset wrapper for over-sampling.

    Parameters:
        dataset
        desired_dist: The desired class distribution. The keys are the classes whilst the
            values are the desired class percentages. The values are normalised so that sum up
            to 1.
        buffer_size: Size of the buffer.
        seed: Random seed for reproducibility.

    Attributes:
        actual_dist: The counts of the observed sample labels.
        rng: A random number generator instance.

    """

    def __init__(self, dataset: torch.utils.data.IterableDataset, desired_dist: dict,
                 buffer_size: int = None, seed: int = None):

        self.dataset = dataset
        self.desired_dist = {c: p / sum(desired_dist.values()) for c, p in desired_dist.items()}
        self.buffer_size = buffer_size
        self.seed = seed

        self.actual_dist = collections.Counter()
        self.rng = random.Random(seed)
        self._pivot = None
        self._buffer = {c: list() for c in desired_dist}

    def __iter__(self):

        for x, y in self.dataset:

            self.actual_dist[y] += 1

            # To ease notation
            f = self.desired_dist
            g = self.actual_dist

            # Add to buffer
            if self.buffer_size is not None:
                self._buffer[y].append(x)
                self._buffer[y] = self._buffer[y][-self.buffer_size:]

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
                if self.buffer_size is None:
                    yield x, y
                else:
                    yield self.rng.choice(self._buffer[y]), y

    @classmethod
    def expected_size(cls, n, desired_dist, actual_dist):
        M = max(
            actual_dist.get(k) / desired_dist.get(k)
            for k in set(desired_dist) | set(actual_dist)
        )
        return int(n * M)

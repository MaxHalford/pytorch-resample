import collections
import random

import torch

from . import utils


class HybridSampler(torch.utils.data.IterableDataset):
    """Dataset wrapper that uses both under-sampling and over-sampling.

    Parameters:
        dataset
        desired_dist: The desired class distribution. The keys are the classes whilst the
            values are the desired class percentages. The values are normalised so that sum up
            to 1.
        sampling_rate: The fraction of data to use.
        buffer_size: Size of the buffer.
        seed: Random seed for reproducibility.

    Attributes:
        actual_dist: The counts of the observed sample labels.
        rng: A random number generator instance.

    """

    def __init__(self, dataset: torch.utils.data.IterableDataset, desired_dist: dict,
                 sampling_rate: float, buffer_size: int = None, seed: int = None):

        self.dataset = dataset
        self.desired_dist = {c: p / sum(desired_dist.values()) for c, p in desired_dist.items()}
        self.sampling_rate = min(max(sampling_rate, 0), 1)
        self.buffer_size = buffer_size
        self.seed = seed

        self.actual_dist = collections.Counter()
        self.rng = random.Random(seed)
        self._n = 0
        self._buffer = {c: list() for c in desired_dist}

    def __iter__(self):

        for x, y in self.dataset:

            self.actual_dist[y] += 1
            self._n += 1

            f = self.desired_dist
            g = self.actual_dist

            # Add to buffer
            if self.buffer_size is not None:
                self._buffer[y].append(x)
                self._buffer[y] = self._buffer[y][-self.buffer_size:]

            rate = self.sampling_rate * f[y] / (g[y] / self._n)

            for _ in range(utils.random_poisson(rate, rng=self.rng)):
                if self.buffer_size is None:
                    yield x, y
                else:
                    yield self.rng.choice(self._buffer[y]), y

    @classmethod
    def expected_size(cls, n, sampling_rate):
        return int(sampling_rate * n)

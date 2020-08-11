# PyTorch resampling methods

## Motivation

[Imbalanced learning](https://www.jeremyjordan.me/imbalanced-data/) is a machine learning paradigm whereby a classifier has to learn from a dataset that has a skewed class distribution. An imbalanced dataset may have a detrimental impact on the classifier's performance.

Rebalancing a dataset is one way to deal with class imbalance. This can be done by:

1. under-sampling common classes.
2. over-sampling rare classes.
3. doing a mix of both.

PyTorch provides [some utilities](https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler) for rebalancing a dataset, but they are limited to batch datasets of known length (i.e., they require a dataset to have a `__len__` method). Community contributions such as [ufoym/imbalanced-dataset-sampler](https://github.com/ufoym/imbalanced-dataset-sampler) are cute, but they also only work with batch datasets (also called *map-style* datasets in PyTorch jargon). There's also a [GitHub issue](https://github.com/pytorch/pytorch/issues/28743) opened on the pytorch/pytorch repository, but it doesn't seem very active.

This repository thus implements data resamplers that work with [`IterableDataset`s](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset), which were added in [this pull request](https://github.com/pytorch/pytorch/pull/19228). In particular, the provided methods do not require you to have to know the size of your dataset in advance. They all work for binary classification as well as multi-class classification.

## Installation

```sh
$ pip install pytorch_resample
```

## Usage

As a running example, we'll define an `IterableDataset` that iterates over the output of scikit-learn's [`make_classification`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html) function.

```py
>>> from sklearn import datasets
>>> import torch

>>> class MakeClassificationStream(torch.utils.data.IterableDataset):
...
...     def __init__(self, *args, **kwargs):
...         self.X, self.y = datasets.make_classification(*args, **kwargs)
...
...     def __iter__(self):
...         yield from iter(zip(self.X, self.y))

```

The above dataset can be provided to a [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) in order to iterate over [`Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) batches. For the sake of example, we'll generate 10.000 samples, with 50% of 0s, 40% of 1s, and 10% of 2s. We can use a [`collections.Counter`](https://docs.python.org/3/library/collections.html#collections.Counter) to measure the effective class distribution.

```py
>>> import collections

>>> dataset = MakeClassificationStream(
...     n_samples=10_000,
...     n_classes=3,
...     n_informative=6,
...     weights=[.5, .4, .1],
...     random_state=42
... )

>>> y_dist = collections.Counter()

>>> batches = torch.utils.data.DataLoader(dataset, batch_size=16)
>>> for xb, yb in batches:
...     y_dist.update(yb.numpy())

>>> for label in sorted(y_dist):
...     print(f'• {label}: {y_dist[label] / sum(y_dist.values()):.2%} ({y_dist[label]})')
• 0: 49.95% (4995)
• 1: 39.88% (3988)
• 2: 10.17% (1017)

```

The data stream can be under-sampled with the `pytorch_resample.UnderSampler` class. The latter is a wrapper that has to be provided with an `IterableDataset` and a desired class distribution. It inherits from `IterableDataset`, and may thus be used instead of the wrapped dataset. As an example, let's make it so that the classes are equally represented.

```py
>>> import pytorch_resample

>>> sample = pytorch_resample.UnderSampler(
...     dataset=dataset,
...     desired_dist={0: .33, 1: .33, 2: .33},
...     seed=42
... )

>>> y_dist = collections.Counter()

>>> batches = torch.utils.data.DataLoader(sample, batch_size=16)
>>> for xb, yb in batches:
...     y_dist.update(yb.numpy())

>>> for label in sorted(y_dist):
...     print(f'• {label}: {y_dist[label] / sum(y_dist.values()):.2%} ({y_dist[label]})')
• 0: 33.30% (1007)
• 1: 33.10% (1001)
• 2: 33.60% (1016)

```

As shown, the observed class distribution is close to the specified distribution. Indeed, there are less 0s and 1s than above. Note that the values of the `desired_dist` parameter are not required to sum up to 1, as this is done for you automatically.

You may use `pytorch_resample.OverSampler` to instead oversample the data. It has the same signature as `pytorch_resample.UnderSampler`, and can thus be used in the exact same manner.

```py
>>> sample = pytorch_resample.OverSampler(
...     dataset=dataset,
...     desired_dist={0: .33, 1: .33, 2: .33},
...     seed=42
... )

>>> y_dist = collections.Counter()

>>> batches = torch.utils.data.DataLoader(sample, batch_size=16)
>>> for xb, yb in batches:
...     y_dist.update(yb.numpy())

>>> for label in sorted(y_dist):
...     print(f'• {label}: {y_dist[label] / sum(y_dist.values()):.2%} ({y_dist[label]})')
• 0: 33.21% (4995)
• 1: 33.01% (4965)
• 2: 33.78% (5080)

```

In this case, the 1s and 2s have been oversampled.

The `pytorch_resample.HybridSampler` class can be used to compromise between under-sampling and over-sampling. It accepts an extra parameter called `sampling_rate`, which determines the percentage of data to use. This allows to control how much data is used for training, whilst ensuring that the class distribution follows the desired distribution.

```py
>>> sample = pytorch_resample.HybridSampler(
...     dataset=dataset,
...     desired_dist={0: .33, 1: .33, 2: .33},
...     sampling_rate=.5,  # use 50% of the dataset
...     seed=42
... )

>>> y_dist = collections.Counter()

>>> batches = torch.utils.data.DataLoader(sample, batch_size=16)
>>> for xb, yb in batches:
...     y_dist.update(yb.numpy())

>>> for label in sorted(y_dist):
...     print(f'• {label}: {y_dist[label] / sum(y_dist.values()):.2%} ({y_dist[label]})')
• 0: 33.01% (1672)
• 1: 32.91% (1667)
• 2: 34.08% (1726)

```

As can be observed, the amount of streamed samples is close to 5000, which is half the size of the dataset.

## Expected number of samples

It's possible to determine the exact number of samples each resampler will stream back in advance, provided the class distribution of the data is known.

```py
>>> n = 10_000
>>> desired = {'cat': 1 / 3, 'mouse': 1 / 3, 'dog': 1 / 3}
>>> actual = {'cat': .5, 'mouse': .4, 'dog': .1}

>>> pytorch_resample.UnderSampler.expected_size(n, desired, actual)
3000

>>> pytorch_resample.OverSampler.expected_size(n, desired, actual)
15000

>>> pytorch_resample.HybridSampler.expected_size(n, .5)
5000

```

## Performance tip

By design `UnderSampler` and `HybridSampler` yield repeated samples one after the other. This might not be ideal, as it is usually desirable to diversify the samples within each batch. We therefore recommend that you use a [shuffling buffer](https://www.moderndescartes.com/essays/shuffle_viz/), such as the `ShuffleDataset` class proposed [here](https://discuss.pytorch.org/t/how-to-shuffle-an-iterable-dataset/64130/6).

## Development

```sh
$ python -m venv .env
$ source .env/bin/activate
$ pip install poetry
$ poetry install
$ pytest
```

## License

The MIT License (MIT). Please see the [license file](LICENSE) for more information.

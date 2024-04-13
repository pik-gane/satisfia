#!/usr/bin/env python3

import math
import random

import numpy as np
import torch

"""Each distribution derives from the base class _distribution. The base class implements all necessary methods except for the constructor and a method for sampling an element from the distribution (_sample_single). At minimum, the derived class must implement those. The derived class can override additional methods to get more accurate and faster implementations for the specific distribution in question (e.g. exact expected value instead of estimation from sampling; population variance instead of sample variance)."""


class _distribution:
    def __init__(self):
        raise NotImplementedError

    def _sample_single(self):
        raise NotImplementedError

    def sample(self, n=1):
        # return torch.tensor([self._sample_single() for _ in range(n)])
        return [self._sample_single() for _ in range(n)]

    def median(self, *, precision=64):
        samples = sorted(self.sample(precision))

        left = (precision - 1) // 2
        right = precision // 2

        return samples[left : right + 1]

    def E(self, *, precision=64):
        samples = self.sample(precision)
        return sum(samples) / len(samples)

    mean = E

    def var(self, *, precision=64):
        samples = self.sample(precision)
        E = sum(samples) / len(samples)
        return sum((samples - E) ** 2) / (precision - 1)  # sample variance

    def stddev(self, *, precision=64):
        return torch.sqrt(self.var(precision))


class categorical(_distribution):
    def __init__(self, a, b=None):
        """If both a and b are specified, they are lists with category names and weights respectively.
        If only a is specified, it's a dictionary of {category name: weight}."""

        categories = a if (b == None) else {name: weight for name, weight in zip(a, b)}

        for name in categories:
            if categories[name] <= 0:
                raise ValueError("Invalid category weight")

        self._category2weight = categories.copy()

        self._weight_total = sum([categories[name] for name in categories])
        self._order_sticky = True

    def category_set(self, name, weight):
        if weight <= 0:
            raise ValueError("Invalid category weight")

        if name in self._category2weight:
            self._weight_total -= self._category2weight[name]
        self._weight_total += weight
        self._order_sticky = True

        self._category2weight[name] = weight

    def category_del(self, name):
        if name in self._category2weight:
            self._weight_total -= self._category2weight[name]
        self._order_sticky = True

        del self._category2weight[name]

    def _select(self, chance):
        def priority(name):
            return -self._category2weight[name]

        if len(self._category2weight) == 0:
            raise ValueError("No categories")

        if self._order_sticky:
            self._order_sticky = False

            self._order = sorted(self._category2weight.keys(), key=priority)

        # Iterate from most to the least probable to reduce expected time.
        # Probability is distributed proportionally to weight.
        for name in self._order:
            weight = self._category2weight[name]
            chance -= weight
            if chance < 0:
                return name

        return self._category2weight[-1]  # this line is reachable due to precision errors

    def _sample_single(self):
        return self._select(random.uniform(0, self._weight_total))

    def median(self):
        # TODO this may return only one of the medians
        return [self._select(self._weight_total / 2.0)]

    def E(self):
        return (
            sum(
                [
                    float(name) * self._category2weight[name]
                    for name in self._category2weight
                ]
            )
            / self._weight_total
        )

    def expectation(self, f, additional_args=()):
        """Return the expected value of f(x, *additional_args) for x ~ this distribution."""
        return (
            np.sum(
                [
                    weight * f(name, *additional_args)
                    for name, weight in self._category2weight.items()
                ],
                axis=0,
            )
            / self._weight_total
        )

    def expectation_of_fct_of_probability(self, f, additional_args=()):
        """Return the expected value of f(x, probability(x), *additional_args) for x ~ this distribution."""
        return (
            np.sum(
                [
                    weight * f(name, weight / self._weight_total, *additional_args)
                    for name, weight in self._category2weight.items()
                ],
                axis=0,
            )
            / self._weight_total
        )

    def var(self):
        E = self.E()
        moment2 = (
            sum(
                [
                    float(name) ** 2 * self._category2weight[name]
                    for name in self._category2weight
                ]
            )
            / self._weight_total
        )
        return moment2 - E**2  # population variance

    def support(self):
        return self._category2weight.keys()

    def score(self, name):
        return math.log(self._category2weight[name] / self._weight_total)

    def probability(self, name):
        return self._category2weight[name] / self._weight_total

    def categories(self):
        for category in self._category2weight:
            yield (category, self._category2weight[category] / self._weight_total)


class uniform_discrete(_distribution):
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self._count = self.high - self.low + 1

    def _sample_single(self):
        return random.randint(self.low, self.high)

    def median(self):
        left = self.low + (self._count - 1) // 2
        right = self.low + (self._count // 2)

        return [left, right] if (right > left) else [left]

    def E(self):
        return (self.low + self.high) / 2

    def var(self):
        return (self._count**2 - 1) / 12

    def support(self):
        return list(range(self.low, self.high + 1))

    def score(self, _):
        return math.log(1 / self._count)


def bernoulli(p):
    return categorical({0: 1 - p, 1: p})


def infer(sample_single):
    class inferred(_distribution):
        def __init__(self):
            pass

        def _sample_single(self):
            return sample_single()

    return inferred()


import unittest


class TestDistributions(unittest.TestCase):
    def test_bernoulli(self):
        b = bernoulli(0)
        for i in range(100):
            self.assertEqual(b.sample(), 0)

        b = bernoulli(1)
        for i in range(100):
            self.assertEqual(b.sample(), 1)

        b = bernoulli(0.75)
        self.assertEqual(b.median()[0], 1)
        self.assertEqual(b.E(), 0.75)
        self.assertEqual(b.var(), 0.1875)

    def test_categorical(self):
        c = categorical({0: 1, 1: 2, 2: 4})
        self.assertEqual(c.median()[0], 2)
        self.assertAlmostEqual(c.E(), 10 / 7, places=5)

    def test_uniform(self):
        die = uniform_discrete(1, 6)
        self.assertEqual(tuple(die.median()), (3, 4))
        self.assertEqual(die.E(), 3.5)
        self.assertAlmostEqual(die.var(), 2.916667, places=5)

        die7 = uniform_discrete(1, 7)
        self.assertEqual(tuple(die7.median()), (4,))
        self.assertEqual(die7.E(), 4)

    def test_infer(self):
        values = [1, 3, 5]
        i = infer(lambda: random.choice(values))
        for s in i.sample(100):
            self.assertIn(s, values)


if __name__ == "__main__":
    unittest.main()

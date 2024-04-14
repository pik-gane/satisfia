#!/usr/bin/env python3

from typing import NamedTuple

def Interval(left, right=None):
    if hasattr(left, "__len__"):
        assert right is None
        return _Interval(*left)
    else:
        return _Interval(left,  left if right is None else right)

class _Interval(NamedTuple):
	left: float
	right: float

	def __str__(self):
		if self.left > self.right:
			return "<empty>"
		return "[{}, {}]".format(self.left, self.right)

	def __repr__(self):
		return "Interval({}, {})".format(self.left, self.right)

	def __contains__(self, item):
		return self.left <= item <= self.right

	def __le__(self, other):
		return other[0] <= self.left <= self.right <= other[1]

	def __eq__(self, other):
		return (self.left == other[0]) and (self.right == other[1])

	def __and__(self, other):
		return _Interval(max(self.left, other[0]), min(self.right, other[1]))

def interpolate(x, l, y):
    # denoted	x : l : y in formulas
    diff = y - x
    return x + l * diff

def interpolate2(x, l, y):
    diff = y - x
    return ( x + l[0] * diff, x + l[1] * diff )

def interpolateInt(x, l, y):
    # one argument is an interval, so everything becomes an interval
    x, l, y = Interval(x), Interval(l), Interval(y)
    return Interval(x[0] + l[0] * (y[0] - x[0]), x[1] + l[1] * (y[1] - x[1]))

def relativePosition2(x, z, y):
    if (diff := (y-x)) == 0.:
        return (.5, .5)

    return ( (z[0] - x) / diff, (z[1] - x) / diff )

def relativePosition(x, z, y):
	# denoted	x \ z \ y in formulas
    if (diff := (y-x)) == 0.:
        return .5

    return (z - x) / diff

def relativePositionInt(x, z, y):
    x, y, z = Interval(x), Interval(y), Interval(z)
    return Interval((z[0] - x[0]) / (y[0] - x[0]) if (y[0] != x[0]) else 0.5,
                (z[1] - x[1]) / (y[1] - x[1]) if (y[1] != x[1]) else 0.5)

def clip2(x, z, y):
	return ( min(max(x, z[0]), y), min(max(x, z[1]), y) )

def clip(x, z, y):
	# denoted	x [ z ] y in formulas
	return min(max(x, z), y)

def clipInt(x, z, y):
    # one argument is an interval, so everything becomes an interval
    x, y, z = Interval(x), Interval(y), Interval(z)
    return Interval(min(max(x[0], z[0]), y[0]), min(max(x[1], z[1]), y[1]))

def between(item, a, b):
	return (a <= item <= b) or (b <= item <= a)

def midpoint(interval):
	return (interval[0] + interval[1]) / 2

def isSubsetOf(interval1, interval2):
	# is interval1 a subset of interval2?
	return (interval2[0] <= interval1[0]) and (interval2[1] >= interval1[1])

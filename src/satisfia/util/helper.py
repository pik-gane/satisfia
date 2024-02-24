#!/usr/bin/env python3

class Interval():
	def __init__(self, left, right=None):
		if hasattr(left, "__len__"):
			if right != None:
				raise TypeError()
			if len(left) != 2:
				raise TypeError()
			self._left = left[0]
			self._right = left[1]
		else:
			self._left = left
			self._right = left if (right == None) else right

	def __str__(self):
		if self._left > self._right:
			return "<empty>"
		return "[{}, {}]".format(self._left, self._right)

	def __repr__(self):
		return "Interval({}, {})".format(self._left, self._right)

	def __contains__(self, item):
		return self._left <= item <= self._right

	def __le__(self, other):
		return other[0] <= self._left <= self._right <= other[1]

	def __eq__(self, other):
		return (self._left == other[0]) and (self._right == other[1])

	def __and__(self, other):
		return Interval(max(self._left, other[0]), min(self._right, other[1]))

	def __getitem__(self, item):
		if item == 0:
			return self._left
		elif item == 1:
			return self._right
		else:
			raise IndexError()

	def __hash__(self):
		return hash((self._left, self._right))

	def __len__(self):
		return 2

def interpolate(x, l, y):
	# denoted	x : l : y in formulas

	if isinstance(x, Interval) or isinstance(l, Interval) or isinstance(y, Interval):
		# one argument is an interval, so everything becomes an interval
		x, l, y = Interval(x), Interval(l), Interval(y)
		return Interval(x[0] + l[0] * (y[0] - x[0]), x[1] + l[1] * (y[1] - x[1]))
	else:
		return x + l * (y - x)

def relativePosition(x, z, y):
	# denoted	x \ l \ y in formulas

	if isinstance(x, Interval) or isinstance(z, Interval) or isinstance(y, Interval):
		# one argument is an interval, so everything becomes an interval
		x, y, z = Interval(x), Interval(y), Interval(z)
		return Interval((z[0] - x[0]) / (y[0] - x[0]) if (y[0] != x[0]) else 0.5,
					(z[1] - x[1]) / (y[1] - x[1]) if (y[1] != x[1]) else 0.5)
	elif x == y:
		return 0.5
	else:
		return (z - x) / (y - x)

	return (l - x) / (y - x)

def clip(x, z, y):
	# denoted	x [ z ] y in formulas

	if isinstance(x, Interval) or isinstance(z, Interval) or isinstance(y, Interval):
		# one argument is an interval, so everything becomes an interval
		x, y, z = Interval(x), Interval(y), Interval(z)
		return Interval(min(max(x[0], z[0]), y[0]), min(max(x[1], z[1]), y[1]))
	else:
		return min(max(x, z), y)

def between(item, a, b):
	return (a <= item <= b) or (b <= item <= a)

def midpoint(interval):
	return (interval[0] + interval[1]) / 2

def isSubsetOf(interval1, interval2):
	# is interval1 a subset of interval2?
	return (interval2[0] <= interval1[0]) and (interval2[1] >= interval1[1])

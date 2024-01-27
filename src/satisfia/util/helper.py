#!/usr/bin/env python3

"""class Interval():
	def __init__(self, left, right=None):
		self.left = left
		self.right = right if (right != None) else left"""

def interpolate(x, l, y):
	# denoted	x : l : y in formulas

	""" TODO implement intervals
	if (_.isArray(x) || _.isArray(lam) || _.isArray(y)) {
		// one argument is an interval, so everything becomes an interval:
		var xx = asInterval(x), lamlam = asInterval(lam), yy = asInterval(y);
		return [xx[0] + lamlam[0] * (yy[0] - xx[0]),
				xx[1] + lamlam[1] * (yy[1] - xx[1])];
	}"""

	return x + l * (y - x);

def between(item, a, b):
	return (a <= item <= b) or (b <= item <= a)

def midpoint(interval):
	return (interval[0] + interval[1]) / 2

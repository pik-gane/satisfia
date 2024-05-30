from scipy.optimize import linprog
import numpy as np


d = 5
n = 100

points = np.random.normal(size=(d, n))
probs = np.random.dirichlet(np.ones(n))
target = np.average(points, axis=1, weights=probs)

A_eq = np.concatenate([points, np.ones((1,n))], axis=0)
b_eq = np.concatenate([target, [1]])
c = np.concatenate([[-1], np.zeros(n-1)])

res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0,1), method='highs')
print(np.round(res.x,7))    
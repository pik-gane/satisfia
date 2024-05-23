import numpy as np
from scipy.spatial import Delaunay
from scipy.special import binom, hyp2f1

## parameters:

ds = [2,3,4,5,6,7,8,9,10]
nits = 1000


# 2^(-n) Sum[Binomial[n, k] Hypergeometric2F1[1, 1 + n, 1 - k + n, 1/2], {k,0,n-1}]:
def sum(n):
    return np.sum([
        binom(n,k) * hyp2f1(1, 1+n, 1-k+n, 1/2)
        for k in range(n)]) / 2**n
print([(d+1,sum(d+1)) for d in ds])

exit()

def simulate(d):
    target = np.repeat(0, d)
    found = False
    vertices = []
    directions = []
    while len(vertices) < 10*d**2 and not found:
        v = np.random.normal(0,1,d)
        v /= np.linalg.norm(v)
        vertices.append(v)
        if len(vertices) >= d+1:
            # check if target is in convex hull of vertices:
            try:
                found = (Delaunay(np.array(vertices)).find_simplex(target) >= 0)
            except:
                pass
        if found: break
        dir = target - v
        dir /= np.linalg.norm(dir)
        directions.append(dir)
        next_direction = np.mean(directions, axis=0)

    return len(vertices)

#simulate(2, 10, plot=True)

# run the simulation N times:
means = np.zeros(len(ds))
maxs = np.zeros(len(ds))
res = []
for i,d in enumerate(ds):
    sims = [simulate(d) for i in range(nits)]
    means[i] = me = np.mean(sims)
    maxs[i] = ma = np.max(sims)
    print("d=%d, mean=%f, max=%f" % (d, me, ma))

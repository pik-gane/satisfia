import numpy as np
import pylab as plt
from scipy.spatial import Delaunay

## parameters:

Ns = range(10,1000,10) #[10,100,1000]
ds = [3]
nits = 1000

def simulate(d, N, plot=False):
    # sample N random policy value points:
    points = np.random.normal(size=(N, d))

    ok = False
    while not ok:
        # sample a random target point:
        target = np.random.normal(size=d)
        # check that the target is in the convex hull:
        try:
            ok = (Delaunay(points).find_simplex(target) >= 0)
        except:
            pass
    # sample a first direction:
    next_direction = np.random.normal(size=d)
    next_direction /= np.linalg.norm(next_direction)
    if plot:
        plt.ion()
        plt.scatter(points[:,0], points[:,1], c='b')
        plt.scatter(target[0], target[1], c='r')
        # show the plot but don't wait for user:
        plt.show()
        plt.pause(0.01)
    found = False
    vertices = []
    directions = []
    while len(vertices)<N and not found:
        # find the policy value point for which the angle between the vector from the target point to that point with the next_direction is smallest:
        scalar_products = np.array([np.dot(next_direction, (p - target)/np.linalg.norm(p - target)) for p in points])
        v = points[np.argmax(scalar_products)]
        vertices.append(v)
        if plot:
            plt.plot([target[0], target[0]+next_direction[0]], [target[1], target[1]+next_direction[1]], c='g')
            plt.scatter(v[0], v[1], c='k')
            plt.draw()
            plt.pause(0.01)
        if len(vertices) >= d+1:
            # check if target is in convex hull of vertices:
            try:
                found = (Delaunay(np.array(vertices)).find_simplex(target) >= 0)
#                poly = polytope.qhull(np.array(vertices))
#                found = target in poly
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
results = {}
res = []
for d in ds:
    for N in Ns:
        results[(d,N)] = r = np.mean([simulate(d,N) for i in range(nits)])
        res.append(r)
        print("d=%d, N=%d, r=%f" % (d, N, r))

print(results)
plt.figure()
plt.plot(res)
plt.show()
plt.pause(100)

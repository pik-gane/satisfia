import numpy as np
import polytope

## parameters:

N = 10 # sample size
max_d = 5

def simulate(d):
    # sample a random target point from a d-dimensional ball:
    target = np.random.normal(size=d)
    target *= np.random.uniform() / np.linalg.norm(target)
    found = False
    points = []
    while not found:
        # sample another point between the unit and radius-two spheres in d-space:
        point = np.random.normal(size=d)
        point *= (1 + np.random.uniform()) / np.linalg.norm(point)
        points.append(point)
        poly = polytope.qhull(np.array(points))
        try:
            found = target in poly
        except:
            pass
        if len(points) >= 30: break
    return len(points)

# run the simulation N times:
for d in range(2, max_d+1):
    results = [simulate(d) for i in range(N)]
    print("Average number of points needed to enclose a target point in", d, "dims:", np.mean(results))
import numpy as np
import pylab as plt
from scipy.spatial import Delaunay


## parameters:

depths = [10]
ds = [2,4,6,8]
nits = 1000

def simulate(dim, depth, only_terminal=0, deterministic=0, plot=False):

    # States are binary encoded as 1sa...sa where each s bit encodes a successor branch 
    # and each a bit an action, read from right to left. is an even number of bits,
    # Actions are binary encoded as 1asa...sa.

    state_level_starts = 1<<(2*np.arange(depth+1))
    state_level_ends = 1<<(2*np.arange(depth+1)+1)
    action_level_starts = 1<<(2*np.arange(depth)+1)
    action_level_ends = 1<<(2*np.arange(depth)+2)
    size = state_level_ends[-1]

    # draw random Delta values for successors and successor=1 probabilities for actions:
    rands = np.random.uniform(size=(size, dim))

    if only_terminal:
        # shift all Delta towards terminal states:
        for level in range(depth-1):
            sls0 = state_level_starts[level+1]
            sle1 = state_level_ends[level+1]
            sle0 = sls1 = (sls0+sle1)//2
            deltas0times4 = np.repeat(rands[sls0:sle0],4,axis=0)
            deltas1times4 = np.repeat(rands[sls1:sle1],4,axis=0)
            w = deltas0times4.shape[0]
            nextsls0 = state_level_starts[level+2]
            rands[nextsls0:nextsls0+w] += deltas0times4
            rands[nextsls0+w:nextsls0+2*w] += deltas1times4
            rands[sls0:sle0] = rands[sls1:sle1] = 0
        aspiration = rands[nextsls0:nextsls0+2*w].mean(axis=0)
    else:
        # set a random aspiration:
        aspiration = (np.random.uniform(size=dim)*0+0.5) * depth

    if plot: plt.ion()

    def get_vertex(aspiration, direction):
        pi = np.zeros(size, dtype=int) # policy: action by state
        Q = np.zeros((size, dim)) # vectors of expected total to go by action
#        Q2 = np.zeros((size, dim, dim)) # matrices of expected raw 2nd moment of total to go by action
        for level in range(depth-1,-1,-1):
            als = action_level_starts[level]
            ale = action_level_ends[level]
            w = ale - als
            psucc1 = rands[als:ale,0] * (0 if deterministic else 1)
            sls0 = state_level_starts[level+1]
            sle0 = sls1 = sls0 + w
            sle1 = state_level_ends[level+1]
            deltas0 = rands[sls0:sle0]
#            deltas0c = deltas0[:,:,None]
#            deltas0r = deltas0[:,None,:]
            deltas1 = rands[sls1:sle1]
#            deltas1c = deltas1[:,:,None]
#            deltas1r = deltas1[:,None,:]
            if level < depth-1:
                nextpi0 = pi[sls0:sle0] # action in successors 0
                nextpi1 = pi[sls1:sle1] # action in successors 1
                nextals0 = action_level_starts[level+1]
                nextals1 = nextals0 + 2*w
                indices0 = np.arange(nextals0,nextals0+w) + nextpi0*w
                indices1 = np.arange(nextals1,nextals1+w) + nextpi1*w
                Vs0 = Q[indices0] # expected total to go in successors 0
                Vs1 = Q[indices1] # expected total to go in successors 1
                thisQ = Q[als:ale] = (1-psucc1)[:,None] * (deltas0 + Vs0) + psucc1[:,None] * (deltas1 + Vs1)
#                thisQ2 = Q2[als:ale] = (
#                    (1-psucc1)[:,None,None] * (deltas0c*deltas0r + deltas0c*Vs0[:,None,:] + Vs0[:,:,None]*deltas0r + Q2[indices0])
#                    + psucc1[:,None,None] * (deltas1c*deltas1r + deltas1c*Vs1[:,None,:] + Vs1[:,:,None]*deltas1r + Q2[indices1])
#                )
            else:
                thisQ = Q[als:ale] = (1-psucc1)[:,None] * deltas0 + psucc1[:,None] * deltas1
#                thisQ2 = Q2[als:ale] = (1-psucc1)[:,None,None] * (deltas0c*deltas0r) + psucc1[:,None,None] * (deltas1c*deltas1r)
            if plot:
                plt.plot(thisQ[:,0],thisQ[:,1],"k.",ms=(depth-level+1)*10, alpha=0.1)
                plt.plot([aspiration[0]],[aspiration[1]],"r.",ms=30)
                plt.show()
                plt.pause(0.01)
            sls = state_level_starts[level]
            sle = state_level_ends[level]
            pi[sls:sle] = np.argmax((
                (thisQ - aspiration) @ direction
                / np.linalg.norm(thisQ - aspiration)
#                / ( np.trace(thisQ2, axis1=1, axis2=2) 
#                    - 2 * thisQ @ aspiration 
#                    + aspiration @ aspiration )**0.5
            ).reshape((2,-1)), axis=0)
        return Q[2+pi[1]]
    
    # sample a first direction:
    next_direction = np.random.normal(size=dim)
    found = False
    vertices = []
    directions = []
    while len(vertices)<10*dim and not found:
        next_direction /= np.linalg.norm(next_direction)
        # find the policy value point for which the angle between the vector from the target point to that point with the next_direction is smallest:
        v = get_vertex(aspiration, next_direction)
        vertices.append(v)
        if plot:
            plt.plot([aspiration[0],v[0]],[aspiration[1],v[1]],"b-")
            next_direction *= np.linalg.norm(v - aspiration)  # only needed for plotting
            plt.plot([aspiration[0],aspiration[0]+next_direction[0]],[aspiration[1],aspiration[1]+next_direction[1]],"r-")
            plt.plot([v[0],aspiration[0]+next_direction[0]],[v[1],aspiration[1]+next_direction[1]],"b--")
            plt.show()
            plt.pause(0.01)
        if len(vertices) >= dim+1:
            # check if target is in convex hull of vertices:
            try:
                found = (Delaunay(np.array(vertices)).find_simplex(aspiration) >= 0)
            except:
                pass
        if found: break
        dir = aspiration - v
        dir /= np.linalg.norm(dir)
        directions.append(dir)
        next_direction = np.mean(directions, axis=0) #+ np.random.normal(size=dim) * 1e-2
    if plot:
        plt.show()
        plt.pause(1000)

    return len(vertices)

means = np.zeros((len(ds), len(depths)))
maxs = np.zeros((len(ds), len(depths)))
res = []
for i,dim in enumerate(ds):
    for j,depth in enumerate(depths):
        sims = [simulate(dim,depth) for i in range(nits)]
        means[i,j] = me = np.mean(sims)
        maxs[i,j] = ma = np.max(sims)
        print("dim=%d, depth=%d, mean=%f, max=%f" % (dim, depth, me, ma))

print(means)
print(maxs)

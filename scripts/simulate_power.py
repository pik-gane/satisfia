from xml.etree.ElementPath import ops
import numpy as np


# parameters:

n_states = 1000
n_players = 10
n_actions = 10
avg_n_successors = 10
iterations = 1000
horizon = 1000
eps = 0.9

# generate random max. transition probs:
psas = np.random.rand(n_states, n_actions, n_states)
psas = np.minimum(0, psas/np.max(psas, axis=2, keepdims=True) - 1 + avg_n_successors/n_states)
psas /= np.sum(psas, axis=2, keepdims=True)
# make sure that under action 0 the current state remains with probability 1-eps:
psas2 = psas * 1
psas2[:,0,:] = 0
psas2[np.arange(n_states),0,np.arange(n_states)] = 1
psas = (1-eps)*psas + eps*psas2
pss = np.max(psas, axis=1)
pmax = np.max(np.sum(pss, axis=1))

# assign states to random players:
player_shares = np.random.rand(n_players)
player_shares /= np.sum(player_shares)
state2player = np.random.choice(list(range(n_players)), p=player_shares, size=n_states, replace=True)

logx = np.zeros((n_states, n_players))
for it in range(iterations):
    newlogx = pss @ logx
    for state in range(n_states):
        player = state2player[state]
        maxlogx = np.max(logx[:,player])
        newlogx[state,player] = maxlogx + np.log(np.dot(pss[state,:], np.exp(logx[:,player] - maxlogx)) / pmax)
    rms = np.sqrt(np.mean((np.exp(logx)-np.exp(newlogx))**2))
    logx = newlogx
    print(it, rms, np.mean(logx, axis=0), np.min(logx))
    if rms < 1e-10: break
print(player_shares)
print(np.sort(logx[:,0]))

logxmax = np.max(logx, axis=1)
print(np.exp(np.sort(logxmax)))

state = 0
vals = []
for t in range(horizon):
    player = state2player[state]
    action = np.argmin(psas[state,:,:] @ logxmax)
    successor = np.random.choice(list(range(n_states)), p=psas[state,action,:])
    print(t, state, player, action, successor, np.exp(logxmax[successor]))
    vals.append(logxmax[successor])
    state = successor

# plot vals:
import matplotlib.pyplot as plt
plt.plot(vals)
plt.show()
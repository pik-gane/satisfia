import numpy as np


# parameters:

n_states = 1000
n_players = 2
n_actions = 5
avg_n_successors = 3
gamma = 1
iterations = 1000
horizon = 20

# generate random max. transition probs:
psas = np.random.rand(n_states, n_actions, n_states)
psas = np.minimum(0, psas/np.max(psas, axis=2, keepdims=True) - 1 + avg_n_successors/n_states)
psas /= np.sum(psas, axis=2, keepdims=True)
pss = np.max(psas, axis=1)
pmax = np.max(np.sum(pss, axis=1))

# assign states to random players:
player_shares = np.random.rand(n_players)
player_shares /= np.sum(player_shares)
state2player = np.random.choice(list(range(n_players)), p=player_shares, size=n_states, replace=True)

x = np.ones((n_states, n_players))
for it in range(iterations):
    newx = np.exp(pss @ np.nan_to_num(np.log(x)) / pmax)
    for state in range(n_states):
        player = state2player[state]
        newx[state,player] = np.dot(pss[state,:], (1-gamma) + gamma*x[:,player]) / pmax
    rms = np.sqrt(np.mean((x-newx)**2))
    x = newx
    print(it, rms, np.mean(x, axis=0), np.min(x))
    if rms < 1e-10: break
print(player_shares)
print(np.sort(x[:,0]))

xmax = np.max(x, axis=1)
print(np.sort(xmax))

state = 0
for t in range(horizon):
    player = state2player[state]
    action = np.argmin(psas[state,:,:] @ xmax)
    successor = np.random.choice(list(range(n_states)), p=psas[state,action,:])
    print(t, state, player, action, successor, xmax[successor])
    state = successor

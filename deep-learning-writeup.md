# Satisfia algorithms with deep reinforcement learning
We want to make be able to run satisfia algorithms on environments which aren't simple gridworlds where planning is not tractable. For this, we train a neural network to predict all the quantities which are computed recursively in planning algorithm. Those quantities can then be used deduce the satisfia policy by inheriting from `AspirationAgent` and overriding the methods which compute those quantities with methods which use the neural network. This is done by the class `AgentMDPDQN`.

## Output of the network
The quantities to which we fit a neural network are:
- `maxAdmissibleQ`, `minAdmissibleQ` (or $\overline{Q}, \underline{Q}$) - the Q values of the maximizer and minimizer. Note: the satisfia algorithm has hyperparameters $\overline\lambda, \underline\lambda$ which we set to $0$ and $1$. For other values of $\overline\lambda, \underline\lambda$, $\overline{Q}, \underline{Q}$ are not the Q values of the maximizer and minimizer anymore and are defined by the following, where $s$ is always a state and $a$ always an action, maxima and minima are taken over the set of all possible actions, $0 \le \gamma \le 1$ is the discount rate, and $\mathbb{E}_{(\delta, s')\sim\mathrm{step}(s, a)}$ means the expectation when $\delta$ and $s'$ are drawn from the reward - next action distribution after taking action $a$ in state $s$. Note how we obtain the classical bellman equations for the Q and V values of the maximizer and minimizer when $(\underline\lambda, \overline\lambda) = (0, 1)$.
$$\overline{V}(s) := \min_a \underline{Q}(s, a) : \overline\lambda : max_a \overline{Q}(s, a)$$
$$\underline{V}(s) := \min_a \underline{Q}(s, a) : \underline\lambda : max_a \overline{Q}(s, a)$$
$$\overline{Q}(s, a) := \mathbb{E}_{(\delta, s')\sim\mathrm{step}(s, a)} (\delta + \gamma\overline{V}(s'))$$
$$\underline{Q}(s, a) := \mathbb{E}_{(\delta, s')\sim\mathrm{step}(s, a)} (\delta + \gamma\underline{V}(s'))$$
where the notation $x:\lambda:y$ stands for interpolation, that is, $x:\lambda:y = x + \lambda(y - x) = (1 - \lambda) x + \lambda y$.

- The safety criteria for which bellman equations exist. An example of a safety criterion for which a bellman formula exists is the safety criterion `Q`, which is the Q value of the actual satisfia policy (that is, `Q(s, a)` is the expected sum of the rewards which one would get if one took action `a` in state `s` and then followed the satisfia policy until the end of the episode). Note that this is different from `maxAdmissibleQ` and `minAdmissibleQ`, which are (when $(\underline\lambda, \overline\lambda) = (0, 1)$) the Q values of the maximizer and minimezer policy. Similar safety criteria are `Q2, Q3, ...`, defined by `Q2(s, a)` being the expected value of the square of the the sum of the rewards which one would get if one took action `a` in state `s` and then followed the satisfia policy until the end of the episode. `Q3, ...` are like `Q2` but with higher powers instead of the square. Note that `Q2(s, a) != (Q(s, a))^2` since the square of an expected value does not equal the expected value of the square. An example of safety criteria for which no bellman formula exists is safety criteria using the Wasserstein distance.

## Architecture of the network
Note that some outputs of the network do not depend on the aspiration and hyperparameters to the satisfia algorithm (i.e. $\overline\lambda, \underline\lambda$, and the weights given to the safety criteria), whereas some outputs of the network do depend on them. For instance, `maxAdmissibleQ` and `minAdmissibleQ` only depend on $\overline\lambda$, $\underline\lambda$ (and we fix $(\overline\lambda, \underline\lambda) = (0, 1)$ for now). In effet, with $(\overline\lambda, \underline\lambda) = (0, 1)$, `maxAdmissibleQ` and `minAdmissibleQ` are the Q values of the maximizer and minimizer, so they don't depend on the aspiration and hyperparameters to the satisfia policy. However, `Q` depends on the aspiration and hyperparameters to the satisfia policy, since it is the Q values of the satisfia policy. The same goes for `Q3, ...`.

Because of this, we decompose the network into multiple parts $f_\theta, g_\phi, h_\psi$. We train the the network by fitting $g_\phi(f_\theta(s))$ to `(maxAdmissibleQ(s, .), minAdmissibleQ(s, .))` and $h_\psi(f_\theta(s), \aleph)$ to `Q(s, .)` (or `(Q(s, .), Q2(s, .), Q3(s, .), ...`), where $s$ is a state and $\aleph$ is an aspiration. Thus, we have a network which has an output not dependent on the aspiration and an output dependent on the aspiration, without the cost of training two networks. Note that the outputs of the network are Q tables, that is, vectors of the form `[maxAdmissibleQ(s, a) for a in all_actions]` (replacing `maxAdmissibleQ` by whatever). Hence the notation `maxAdmissibleQ(s, .)`.

For now, $f_\theta, g_\phi, h_\psi$ are fully connected networks with dropout and layer normalization.

## Training

We use the [DQN algorithm](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).

We could learn this algorithm to learn `maxAdmissibleQ` and `minAdmissibleQ`, just by applying it to train a maximizer and a minimizer. However, this doesn't work that well because there are some states which the maximizer or minimizer never visit, but which the satisfia policy visits. Thus, the Q values in this these states are not learned correctly, which doesn't matter for the maximizer and minimizer, but matters for the satisfia algorithm and makes it not work.

To mitigate this problem, we notice that DQN need not follow the maximizer or minimizer policy in order to learn the maximizer's and minimizer's Q values (this is a well known fact about DQN in classical settings: even though DQN explores using an $\epsilon$-greedy policy where it takes a random action with probability $\epsilon$ at each step, it learns the Q values of the maximizer, not of the $\epsilon$-greedy policy). So we explore using the satisfia policy during training (we also take a random action instead of the satisfia policy's action with some probability, like in $\epsilon$-greedy exploration). Note that to explore using the satisfia policy requires to have an aspiration, which we draw randomly at the beginning of each episode.

In order to learn safety criteria, we do the same thing as DQN but with a different bellman equation.

Note that it is slow to evaluate the satisfia policy at every step when exploring because it is not vectorized, so we reimplement a part of the satisfia algorithm in `src/satisfia/agents/learning/dqn/agent_mdp_dqn.py` in a vectorized manner.

## Environments

For now, we use simple gridworlds for debugging because things don't quite work yet and gridwords require less compute and we have access to the ground truth Q values and safety criteria.

It should, in theory, be easy to make the learning algorithm work with any environment which is a `gymnasium.Env`, since the learning algorithm only uses the `gymnasium.Env` interface and doesn't use anything specific to the gridworlds.

There is a list of environments where misalignment happened, usually without people expecting it to happen, in `environment-compilation.md`. I think it would be really great if we were able to run the satisfia deep learning algorithm on some of them and see if they mitigate the misalignment. I tried to sort them in order of relevance to satisfia, but it was some time ago and I now disagree with my ranking in some cases. I think the best thing to start with is environments of the type "a robot in a physics simulation exploits a big in the psysics engine" or "an agent exploits a weird balance in a video game", with a slight preference for the former because there are more such environments in the list.

## Code structure

The main algorithm is in the `train_dqn` function in `src/satisfia/agents/learning/dqn/train.py`. This is the most important function to understand. All the other files relevant to the train algorithm are in `src/satisfia/agents/learning` and `train_dqn` calls into them. To run the code, run `python scripts/test_dqn.py` in the root directory. `scripts/test_dqn.py` runs the satisfia deep learning algorithm on simple gridworlds and plot some stuff, namely, the evolution of the outputs of the networks during training and an achieved total vs aspiration graph (on which we should see an `x = clamp(y, max=..., min=...)` line if everything is working).
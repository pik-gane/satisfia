import pyspiel
from world_model import MDPWorldModel
from satisfia.agents.makeMDPAgentSatisfia import AgentMDPPlanning
from collections import defaultdict

class FictitiousPlayMDP(MDPWorldModel):

    def __init__(self, *args, game=None, this_player=None, **kwargs):
        self.game = game
        self.this_player = this_player  # which player's perspective we're taking
        super().__init__(*args, **kwargs)
        self.past_others_action_profiles_by_state = defaultdict(defaultdict(int))

    def possible_actions(self, state):
        return self.game.legal_actions(state, self.this_player) # FIXME
    
    def is_terminal(self, state):
        return state.is_terminal()
    
    def transition_distribution(self, state, action):
        # get the current state's past others' action profile counts:
        others_action_profiles = self.past_others_action_profiles_by_state[state]
        # normalize these counts to get a mixed strategy for the other players:
        total = sum(others_action_profiles.values())
        others_mixed_strategy = {action_profile: count / total for action_profile, count in others_action_profiles.items()}
        # construct the transition distribution as a mixture distribution of the other players' mixed strategies:
        transition_distribution = defaultdict(float)
        for others_action_profile, prob in others_mixed_strategy.items():
            # compile the full action profile from action and others_action_profile:
            action_profile = [action if i == self.this_player 
                              else others_action_profile[i if i<self.this_player else i-1]
                              for i in range(self.game.num_players())]
            # ask the game what the distribution of next states would be if the other players took the given action profile and I was to take the given action:
            # TODO! something like this_profiles_transition_distribution = this.game.transitions(state, action_profile)
            # add this_profiles_transition_distribution to transition_distribution, weighted by prob
            for new_state in this_profiles_transition_distribution:
                transition_distribution[new_state] += prob * this_profiles_transition_distribution[new_state]
        return { new_state: (prob, True) for new_state, prob in transition_distribution.items() }
    
    def observation_and_reward_distribution(self, state, action, next_state):
        # TODO: ask the game for the observation and reward given this next_state: delta = ...
        return { (next_state, delta): (1, False) }
    
    def update(self, state, action_profile):
        # update the model based on the action_profile taken in the given state
        others_action_profile = [action for i, action in enumerate(action_profile) if i != self.this_player]
        # TODO: can states be hashed?
        self.past_others_action_profiles_by_state[state][others_action_profile] += 1



class Player:
    
    def __init__(self, game, player_id):
        self.game = game
        self.player_id = player_id

    def reset(self, initial_state):
        self.last_action_profile = None
        self.state = initial_state
        self.current_action = None

    def act(self): raise NotImplementedError()

    def observe(self, action_profile, new_state):
        self.last_action_profile = action_profile
        self.state = new_state

class QLearningPlayer(Player):
    pass
    # here we might wrap pyspiel's standard implementation of Q learning

class SatisfiaPlayer(Player):
    # here we'll wrap AgentMDPPlanning plus stuff for fictitious play

    def __init__(self, game, player_id, initial_aspiration):
        super().__init__(game, player_id)
        self.world_model = FictitiousPlayMDP(game=game, this_player=player_id)
        self.agent = AgentMDPPlanning(params, world=self.world_model)
        self.initial_aspiration = initial_aspiration
        self.state = None
        self.aspiration = None
        self.current_action = None
        self.action_aspiration = None
        self.last_action_profile = None

    def reset(self, initial_state):
        super().reset(initial_state)
        self.world_model.reset(initial_state)
        self.aspiration = self.initial_aspiration

    def act(self):
        local_policy = self.agent.localPolicy(self.state, self.aspiration)
        self.current_action, self.action_aspiration = local_policy.sample()[0]
        self.last_action_profile = None
        return self.current_action
    
    def observe(self, action_profile, new_state):
        self.world_model.update(self.state, action_profile)
        super().observe(action_profile, new_state)
        # TODO: figure out how to deal with rewards/delta
        self.aspiration = self.agent.propagateAspiration(self.state, self.current_action, self.action_aspiration, state.rewards, new_state)
        self.current_action = None


game = pyspiel.load_game("tic_tac_toe")
players = [QLearningPlayer(game, 0), SatisfiaPlayer(game, 1)]

state = game.new_initial_state()
while not state.is_terminal():
    if state.is_simultaneous_node():
        # ask each player for an action:
        action_profile = []
        for player in range(game.num_players()):
            action_profile.append(action)
        # then apply the actions:
        state.apply_actions(action_profile)
        # tell each player what the other player did and what the new state is:
        for player in range(game.num_players()):
            players[player].observe(action_profile, state)
    else:
        raise NotImplementedError()




exit()

# from pyspiel's example.py:

game = pyspiel.load_game(FLAGS.game_string)

# Create the initial state
state = game.new_initial_state()

# Print the initial state
print(str(state))

while not state.is_terminal():
    # The state can be three different types: chance node,
    # simultaneous node, or decision node
    if state.is_chance_node():
        # Chance node: sample an outcome
        outcomes = state.chance_outcomes()
        num_actions = len(outcomes)
        print("Chance node, got " + str(num_actions) + " outcomes")
        action_list, prob_list = zip(*outcomes)
        action = np.random.choice(action_list, p=prob_list)
        print("Sampled outcome: ",
            state.action_to_string(state.current_player(), action))
        state.apply_action(action)
    elif state.is_simultaneous_node():
        # Simultaneous node: sample actions for all players.
        random_choice = lambda a: np.random.choice(a) if a else [0]
        chosen_actions = [
            random_choice(state.legal_actions(pid))
            for pid in range(game.num_players())
        ]
        print("Chosen actions: ", [
            state.action_to_string(pid, action)
            for pid, action in enumerate(chosen_actions)
        ])
        state.apply_actions(chosen_actions)
    else:
        # Decision node: sample action for the single current player
        action = random.choice(state.legal_actions(state.current_player()))
        action_string = state.action_to_string(state.current_player(), action)
        print("Player ", state.current_player(), ", randomly sampled action: ",
            action_string)
        state.apply_action(action)
    print(str(state))

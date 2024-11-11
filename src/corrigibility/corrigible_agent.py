import numpy as np
from extensive_form_game import PerfectInfoExtensiveFormGame


# TODO: add a non-recursive, brute-force whole policy optimization version!


class PerfectInfoCorrigibleAgent:

    world_model: PerfectInfoExtensiveFormGame = None
    """the world model of the agent"""

    agent_player = None
    """the player in the world model played by the agent"""

    principal_players = None
    """a list of the players in the world model acknowledged as the agent's principals"""

    exponent = None
    """probability exponent used in power calculations"""

    def __init__(self, world_model=None, agent_player=None, principal_players=None, exponent=2):
        assert isinstance(world_model, PerfectInfoExtensiveFormGame)
        self.world_model = world_model
        assert all(principal_player in world_model.players for principal_player in principal_players)
        self.principal_players = principal_players
        assert agent_player in world_model.players
        assert agent_player not in principal_players
        self.agent_player = agent_player
        assert exponent >= 1
        self.exponent = exponent

    def plan(self, node):
        """return a continuation policy and the principal power evaluations for the given node"""
        nd = self.world_model.get_node_data(node)
        policy = {}
        powers = {}
        if nd.player == self.agent_player:
            # plan to take the action that maximizes the sum of the principal power evaluations of the corresponding successor node:
            plans = [ self.plan(successor) for successor in nd.successor_ids ]
            values = [ sum(next_powers.values()) for next_policy, next_powers in plans ]
            i = np.argmax(values) 
            action = nd.actions[i]
            next_policy, next_powers = plans[i]
            policy[node] = action
            policy.update(next_policy)
            event = nd.events[i]
            powers[event] = 1
        elif nd.player in self.world_model.probabilistic_players:
            for i, action in enumerate(nd.actions):
                action_probability = nd.probabilities[i]
                next_policy, next_powers = self.plan(nd.successor_ids[i])
                policy.update(next_policy)
                for event, success_probability in next_powers.items():
                    powers[event] = powers.get(event, 0) + action_probability * success_probability
        elif nd.player in self.principal_players:
            for i, action in enumerate(nd.actions):
                next_policy, next_powers = self.plan(nd.successor_ids[i])
                policy.update(next_policy)
                for event, success_probability in next_powers.items():
                    powers[event] = max(powers.get(event, 0), success_probability)  # best-case w.r.t. actions of principal players
        elif nd.player is not None:
            plans = [ self.plan(successor) for successor in nd.successor_ids ]
            # collect all events that might occurr under at least one action:
            events = set()
            for next_policy, next_powers in plans:
                events.update(next_powers.keys())
            # for each event, take the minimum probability across all actions:
            for next_policy, next_powers in plans:
                policy.update(next_policy)
                for event in events:
                    powers[event] = min(powers.get(event, 1), next_powers.get(event, 0))  # worst-case w.r.t. actions of non-probabilistic other players
        else: # terminal node
            pass
        return policy, powers
    
    def evaluate_policy(self, policy):
        """return the sum of the principal power evaluations at all principal decision nodes (!) for the given policy"""
        raise NotImplementedError

    def all_policies(self):
        """return a generator for all possible agent policies"""
        raise NotImplementedError   

if __name__ == "__main__":
    from extensive_form_game import SimpleExample
    game = SimpleExample()
    agent = PerfectInfoCorrigibleAgent(world_model=game, agent_player="Cora", principal_players=["Prince"])
    print(agent.plan(game.initial_node))
import numpy as np
from extensive_form_game import PerfectInfoExtensiveFormGame


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
            plans = [ self.plan(successor) for successor in nd.successor ]
            values = [ sum(next_powers.values()) for next_policy, next_powers in plans ]
            i = np.argmax(values) 
            action = nd.actions[i]
            next_policy, next_powers = plans[i]
            policy[node] = action
            policy.update(next_policy)
            event = nd.event[i]
            powers[event] = 1
        elif nd.player in self.world_model.probabilistic_players:
            for i, action in enumerate(nd.actions):
                action_probability = nd.probability[i]
                next_policy, next_powers = self.plan(nd.successor[i])
                policy.update(next_policy)
                for event, success_probability in next_powers.items():
                    powers[event] = powers.get(event, 0) + action_probability * success_probability
        elif nd.player in self.principal_players:
            for i, action in enumerate(nd.actions):
                next_policy, next_powers = self.plan(nd.successor[i])
                policy.update(next_policy)
                for event, success_probability in next_powers.items():
                    powers[event] = max(powers.get(event, 0), success_probability)  # best-case w.r.t. actions of principal players
        elif nd.player is not None:
            plans = [ self.plan(successor) for successor in nd.successor ]
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
    

if __name__ == "__main__":
    from extensive_form_game import SimpleExample
    game = SimpleExample()
    agent = PerfectInfoCorrigibleAgent(world_model=game, agent_player="Cora", principal_players=["Prince"])
    print(agent.plan(game.initial_node))
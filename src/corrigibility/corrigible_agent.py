import numpy as np
from typing import Tuple

from corrigibility.world_model import AttainmentProbabilities, DiscountRate, NodeId, Policy, Powers, WorldModel


class CorrigibleAgent:

    world_model: WorldModel = None
    """the world model of the agent"""

    power_discount_rate: DiscountRate = None

    boltzmann_temperature: float = None
    """the temperature parameter for the Boltzmann distribution"""

    def __init__(self, world_model=None, power_discount_rate=0.0, boltzmann_temperature=0.0):
        assert isinstance(world_model, WorldModel)
        self.world_model = world_model
        assert power_discount_rate >= 0.0
        self.power_discount_rate = power_discount_rate
        assert boltzmann_temperature >= 0.0
        self.boltzmann_temperature = boltzmann_temperature

    def plan(self, node_id: NodeId, last_principal_node_id: NodeId = None) -> Tuple[Policy, Powers, AttainmentProbabilities]:
        """return a continuation policy for the given node, and the actual principal power evaluations and actual goal attainment probabilities resulting from it"""
        node = self.world_model.node_data(node_id)
        player = node.player
        actions = node.actions

        if node.is_terminal:
            return {}, { node_id: node.actual_power }, {}

        # so this node is not terminal
        agent_policy = {} # Policy()
        actual_powers = {} # Powers()
        attainment_probabilities = {} # AttainmentProbabilities()
        is_principal = player in self.world_model.principal_players
        if is_principal:
            last_principal_node_id = node_id

        # get agent's continuation policy:
        for action in actions:
            successor_id = node.consequences[action]
            later_policy, later_powers, _ = self.plan(successor_id, last_principal_node_id)
            agent_policy.update(later_policy)
            actual_powers.update(later_powers)

        if player == self.world_model.agent_player:
            # calculate and store Boltzmann local policy based on actual powers and durations:
            action_propensities = { 
                action:
                np.exp((actual_powers.get(node.consequences[action], None) or 1) 
                       * np.exp(-self.power_discount_rate * node.durations.get(action, 1)) 
                       / self.boltzmann_temperature)
                for action in actions }
            total_propensities = sum(action_propensities[action] for action in actions)
            node.local_policy = { action: action_propensities[action] / total_propensities 
                                  for action in actions }
            self.world_model.update_nodes({ node_id: node })  # store computed policy in the world model

        # get principal's actual power and actual attainment probabilities,
        # based on actual agent policy rather than principal's beliefs about agent policy
        # (hence using override_policy_belief=True, which works as the previous line has already stored 
        # the agent's continuation policy in the world model):
        _, _, _, powers, attprobs = self.world_model.get_assumed_principal_policy_and_more(node_id, last_principal_node_id, override_policy_belief=True)
        actual_powers.update({ node_id: powers[node_id] })
        attainment_probabilities.update(attprobs)

        return agent_policy, actual_powers, attainment_probabilities

if __name__ == "__main__":
    from corrigibility.world_model import SimpleExample
    game = SimpleExample()
    agent = CorrigibleAgent(world_model=game)
    print(agent.plan(game.initial_node))
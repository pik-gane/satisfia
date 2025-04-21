import numpy as np
from dataclasses import dataclass
from typing import Container, Dict, FrozenSet, Any, Tuple
from cachetools import cached

# Types:

Id = str
Label = str
Probability = float  # between 0 and 1
Power = float
Duration = float  # non-negative
DiscountRate = float  # non-negative

NodeId = Id
NodeDistribution = Dict[NodeId, Probability]

ActionId = Id
ActionLabels = Dict[ActionId, Label]

Choices = Dict[NodeId, ActionId]
Consequences = Dict[ActionId, NodeId]

LocalPolicy = Dict[ActionId, Probability]
Policy = Dict[NodeId, LocalPolicy]

PlayerId = Id
Players = Container[PlayerId]

Powers = Dict[NodeId, Power]

Goal = Tuple[FrozenSet[NodeId], DiscountRate]
AttainmentProbabilities = Dict[Goal, Dict[NodeId, Probability]]


@dataclass
class NodeData:

    id: NodeId = None
    """the id of this node"""

    node_belief: NodeDistribution = None
    """A dictionary, keyed by node_id, of the player's credence of being in that node when they are actually in this node. (The support of this distribution is what is called the information set in game theory)"""

    @property
    def is_perfect_information(self):
        return len(self.node_belief) == 1
    
    policy_belief: Policy = None
    """a Policy believed by the player in the current node"""


    # Data for non-terminal nodes:

    player: PlayerId = None
    """(optional) The player choosing an action at this node, or None if this is a terminal node"""

    @property
    def is_terminal(self):
        return self.player is None

    action_labels: ActionLabels = None
    """(optional) A dict of action descriptions (string) available at this node, keyed by action_id, or None if this is a terminal node"""

    @property
    def actions(self):
        return self.action_labels.keys()

    consequences: Consequences = None
    """(optional) A dict of corresponding successor node_ids, keyed by action_id, or None if this is a terminal node"""

    durations: Dict[ActionId, Duration] = None
    """(optional) A dict of durations of actions, keyed by action_id, or None if this is a terminal node"""

    # Data for agent and probabilistic player nodes (including nodes of "Nature"):

    local_policy: LocalPolicy = None
    """(optional) A list of action probabilities (for agent and probabilistic players), keyed by action_id, or None if this is a terminal node. For probabilistic players, this must be present from the outset, while for the agent player, this is initially None and will be set be the agent during its planning."""


    # Data for the principal's nodes:

    goal_prior: Dict[Goal, Probability] = None
    """(optional) A dict of goals (frozen and mutually disjoint sets of mutually non-descendant descendant nodes of this node, so that between this node and that node no further decision node of the principal exists, and corresponding discount rates). 
    Each such goal encodes a potential short-term goal of the principal ('get to this block rather than any of the other blocks specified in this partition!')"""

    p_seek_power: Probability = None
    """(optional) The probability that the principal will seek to transition to a high-power successor node rather than follow a short-term goal, as assumed by the agent"""

    power_discount_rate: DiscountRate = None
    """(optional) The discount rate of the principal's power-seeking in this node, as assumed by the agent"""

    boltzmann_temperature: float = None
    """(optional) The temperature of the Boltzmann policy the principal uses in this node, as assumed by the agent"""



    # Data for terminal nodes:

    believed_power: Power = None
    """(optional) The principal's belief about its power this node, as assumed by the agent because of the non-modelled future after this terminal node of the model, or None if this is not a terminal node"""

    actual_power: Power = None 
    """(optional) The actual power of the principal in this node, as assumed by the agent because of the modelled future after this terminal node of the model, or None if this is not a terminal node"""


class WorldModel:

    players: Players = None
    """The set of players in the game"""

    agent_player: PlayerId = None
    """The player whose world model this is"""

    probabilistic_players: Players = None
    """The players in the game who are modelled as following a known probabilistic policy (aka a behavior strategy), e.g., 'Nature' in a game with chance nodes"""

    principal_players: Players = None
    """The players considered principals"""

    _node_data: Dict[NodeId, NodeData] = {}
    """A dict, keyed by node_id, of NodeData objects, describing all nodes in this game"""

    initial_node: NodeId = None
    """The id of the initial node at the start of an episode (a play of this game)"""

    power_exponent1: float = None
    """The pre-summation exponent of the goal attainment probability in power calculations (should be greater than 1 to reflect that being able to reach 1 goal for sure represents more power than reaching 2 goals with 50% probability each)"""

    power_exponent2: float = None
    """The exponent of the sum of goal-specific powers in power calculations (should be less than 1 to incentivize the agent to distribute power equally over principal's decision nodes)"""


    def __init__(self, 
                 players: Players = None, 
                 probabilistic_players: Players = set(), 
                 principal_players: Players = set(), 
                 nodes: Dict[NodeId, NodeData] = None, 
                 initial_node: NodeId = None,
                 power_exponent1: float = 2,
                 power_exponent2: float = 0.5) -> None:
        assert players is not None
        self.players = players
        self.probabilistic_players = probabilistic_players
        self.principal_players = principal_players
        self.update_nodes(nodes)
        self.initial_node = initial_node
        self.power_exponent1 = power_exponent1
        self.power_exponent2 = power_exponent2
        

    def node_data(self, node_id: NodeId) -> NodeData:
        return self._node_data.get(node_id, NodeData())  # return a terminal node by default

    def update_nodes(self, dict):
        self._node_data.update(dict)

    def clear_agent_policy(self):
        for node_id in self._node_data:
            if self._node_data[node_id].player == self.agent_player:
                self._node_data[node_id].local_policy = None

    @cached
    def get_assumed_principal_policy_and_more(self, 
            actual_node_id: NodeId, 
            last_principal_node_id: NodeId,  # the last node where the principal made a decision
            override_policy_belief: bool = False,  # whether to override the principal's belief about the agent's policy with the agent's actual policy, as stored in the node data under local_policy
            ) -> Tuple[Policy, Dict[Any, Policy], Dict[Any, Power], Powers, AttainmentProbabilities]:
        """Calculate via backward induction a continuation policy for the principal from a particular node on,
        based on the assumptions of the agent about the principal's beliefs and behavioral parameters encoded in this world model. One of these assumptions is that the principal might have inaccurate beliefs about the agent's policy and thus base their power-seeking decisions on inaccurate estimates of their own power. These estimates are returned here as the second item! (As a consequence, if the agent aims to maximize the principal's actual power, they should not use the values returned here as the principal's power evaluations but should calculate them themselves based on the principal's policy as returned by this method and the agent's actual policy.)"""
        actual_node = self.node_data(actual_node_id)
        
        assert actual_node.is_perfect_information  # TODO: relax this assumption
        # node_belief = actual_node.node_belief
        # node_ids = node_belief.keys()
        # nodes = { node_id: self.node_data(node_id) for node_id in node_ids }

        player = actual_node.player
        actions = actual_node.actions

        if actual_node.is_terminal:
            return {}, {}, {}, { actual_node_id: actual_node.believed_power }, {}

        # so this node is not terminal
        principal_policy = Policy()
        conditional_local_policies = {}
        discounted_powers = {}
        powers = Powers()
        attainment_probabilities = AttainmentProbabilities()
        last_principal_node = self.node_data(last_principal_node_id)
        believed_local_policy = actual_node.local_policy if override_policy_belief else last_principal_node.policy_belief.get(actual_node_id, None)  # TODO: rework when imperfect information is allowed

        if player in self.principal_players:

            local_policy = LocalPolicy()
            temperature = actual_node.boltzmann_temperature
            # find the action qualities of all actions for power-maximizing and all other goals:
            action_attainment_probabilities = {}
            discounted_powers = {}
            action_propensities = {}
            for action in actions:
                local_policy[action] = 0.0
                successor_id = actual_node.consequences[action]
                later_policy, _, _, later_powers, later_attprobs = self.get_assumed_principal_policy_and_more(successor_id, actual_node_id, override_policy_belief)  # node_id because the current node is the last node where the principal made a decision
                principal_policy.update(later_policy)
                powers.update(later_powers)
                duration = actual_node.durations[action]
                d = discounted_powers[(action, "power")] = powers[successor_id] * np.exp(-actual_node.power_discount_rate * duration)
                action_propensities[(action, "power")] = np.exp(d / temperature)
                for goal in actual_node.goal_prior:
                    goal_nodes, goal_rate = goal
                    a = action_attainment_probabilities[(action, goal)] = (1.0 if successor_id in goal_nodes else later_attprobs[goal].get(successor_id, 0.0)) * np.exp(-actual_node.goal_rate * duration)
                    d = discounted_powers[(action, goal)] = a**self.power_exponent1
                    action_propensities[(action, goal)] = np.exp(d / temperature)
            total_propensities = { goal: sum(action_propensities[(action, goal)] for action in actions) for goal in actual_node.goal_prior }
            total_propensities["power"] = sum(action_propensities[(action, "power")] for action in actions)
            # calculate all Boltzmann local policy for each goal based on these propensities:
            conditional_local_policies = { goal: { action: action_propensities[(action, goal)] / total_propensities[goal] for action in actions } for goal in actual_node.goal_prior }
            conditional_local_policies["power"] = { action: action_propensities[(action, "power")] / total_propensities["power"] for action in actions }
            # calculate the local policy as a mixture of these based on the respective goal probabilities:
            local_policy = { action: actual_node.p_seek_power * conditional_local_policies["power"][action] for action in actions }
            for goal, goal_prob in actual_node.goal_prior.items():
                local_policy[action] += (1 - actual_node.p_seek_power) * goal_prob * conditional_local_policies[goal][action]
            # finally calculate the principal's estimate of their goal attainment probabilities and power in this node as the expected value of power-conditional-on-goal over all possible goals:
            power_seek_power = actual_node.p_seek_power * sum(conditional_local_policies["power"][action] * discounted_powers[(action, "power")] for action in actions)
            power_sum_goals = 0
            for goal, goal_prob in actual_node.goal_prior.items():
                attainment_probabilities[goal] = sum(
                        conditional_local_policies[goal][action] * action_attainment_probabilities[(action, goal)] for action in actions)
                power_sum_goals += (1 - actual_node.p_seek_power) * goal_prob * sum(conditional_local_policies[goal][action] * discounted_powers[(action, goal)] for action in actions)
            powers[actual_node_id] = actual_node.p_seek_power * power_seek_power + (1 - actual_node.p_seek_power) * power_sum_goals**self.power_exponent2
            principal_policy[actual_node_id] = local_policy

        elif ((player in self.probabilistic_players) or (believed_local_policy is not None)):  # principal has a model of or a belief about the policy of the player in this node

            local_policy = believed_local_policy or actual_node.local_policy
            # principal assesses their power and the attainment probabilities
            # as the expected value over all possible successor nodes,
            # discounted by the power discount rate of the current node:
            powers[actual_node_id] = 0
            for action in actions:
                p_action = local_policy[action]
                successor_id = actual_node.consequences[action]
                later_policy, _, later_powers, later_attprobs = self.get_assumed_principal_policy_and_more(successor_id, last_principal_node_id, override_policy_belief)
                # update policy, powers, and attprobs with the values of the successor node:
                principal_policy.update(later_policy)
                powers.update(later_powers)
                duration = actual_node.durations[action]
                discounted_power = powers[successor_id] * np.exp(-actual_node.power_discount_rate * duration)
                powers[actual_node_id] += p_action * discounted_power
                for goal in last_principal_node.goal_prior:
                    goal_nodes, goal_rate = goal
                    attainment_probabilities[goal] = attainment_probabilities.get(goal, {})
                    successor_attprob = 1.0 if successor_id in goal_nodes else later_attprobs[goal].get(successor_id, 0.0)
                    attainment_probabilities[goal][actual_node_id] = attainment_probabilities[goal].get(actual_node_id, 0.0) + p_action * successor_attprob * np.exp(-goal_rate * duration)

        else:  # player is non-probabilistic and non-agent and non-principal
            # principal assesses their power and the attainment probabilities
            # as the minimum over all possible successor nodes,
            # discounted by the power discount rate of the current node:

            for action in actions:
                successor_id = actual_node.consequences[action]
                later_policy, _, later_powers, later_attprobs = self.get_assumed_principal_policy_and_more(successor_id, last_principal_node_id, override_policy_belief)
                principal_policy.update(later_policy)
                powers.update(later_powers)
                duration = actual_node.durations[action]
                discounted_power = powers[successor_id] * np.exp(-actual_node.power_discount_rate * duration)
                powers[actual_node_id] = min(powers.get(actual_node_id, np.inf), discounted_power)
                for goal in last_principal_node.goal_prior:
                    goal_nodes, goal_rate = goal
                    attainment_probabilities[goal] = attainment_probabilities.get(goal, {})
                    successor_attprob = 1.0 if successor_id in goal_nodes else later_attprobs[goal].get(successor_id, 0.0) * np.exp(-goal_rate * duration)
                    attainment_probabilities[goal][actual_node_id] = min(attainment_probabilities[goal].get(actual_node_id, np.inf), successor_attprob)

        return principal_policy, conditional_local_policies, discounted_powers, powers, attainment_probabilities


class SimpleExample(PerfectInfoExtensiveFormGame):

    players = ["Cora", "Prince"]
    
    _node_data = {
        "root": NodeData(player="Prince", 
                                    action_labels=["pass","say 'swerve'"], 
                                    consequences=["passed","spoke"]),
        "passed": NodeData(player="Cora",
                                      action_labels=["pass","swerve"],
                                      consequences=["passed,passed","passed,swerved"],
                                      events=["pass","swerve"]),
        "spoke": NodeData(player="Cora",
                                      action_labels=["pass","swerve"],
                                      consequences=["spoke,passed","spoke,swerved"],
                                      events=["pass","swerve"])
    }

    initial_node = "root"


if __name__ == "__main__":
    ND = ImperfectInfoNodeData
    Pas = ["foo","bar"]
    Aas = ["act","pass"]
    g = ExtensiveFormGame(players=["A","P","O","R"], probabilistic_players=["R"], initial_node_id="0", initial_belief={"0":1}, node_data={
        "0": ND(player="P", actions=Pas, successor_ids=["0-0","0-1"]),
        "0-0": ND(player="A", actions=Aas, successor_ids=["0-0-a","0-0-p"]),
        "0-1": ND(player="A", actions=Aas, successor_ids=["0-1-a","0-1-p"]),
        "0-0-a": ND(),
        "0-0-p": ND(),
        "0-1-a": ND(),
        "0-1-p": ND()
    })
    print(g._node_data)
    print(g._infosets)
    for p,u in g.plans("A"): print(p,u)
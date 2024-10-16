from dataclasses import dataclass

@dataclass
class PerfectInfoNodeData:

    player: str = None
    """the player moving at this node"""
    actions: list = None
    """a list of actions available at this node"""
    successor: dict = None
    """a list of corresponding successor nodes"""
    event: dict = None
    """an optional list of corresponding events (for agent players)"""
    probability: dict = None
    """an optional list of probabilities (for probabilistic players)"""


@dataclass 
class ImperfectInfoNodeData(PerfectInfoNodeData):

    infoset: str = None
    """the information set this node belongs to"""


class _BaseGame:

    players: list = None
    """a list of the players in the game"""

    probabilistic_players: list = []
    """a list of players in the game who are modelled as following a known probabilistic policy (aka a behavior strategy)"""


class PerfectInfoExtensiveFormGame(_BaseGame):

    initial_node: str = None
    """the initial node at the start of the episode"""

    def get_node_data(self, node) -> PerfectInfoNodeData:
        return self._node_data.get(node, PerfectInfoNodeData())  # return a terminal node by default


class ExtensiveFormGame(_BaseGame):

    initial_infoset: str = None
    """the initial infoset at the start of the episode"""
    initial_belief: dict = None
    """the initial belief about the initial state (dict of credence by node)"""

    def get_node_data(self, node) -> ImperfectInfoNodeData:
        return self._node_data.get(node, ImperfectInfoNodeData())  # return a terminal node by default
    

    def get_nodes(self, infoset) -> list:
        """return the list of all nodes belonging to the same infoset"""
        return self._nodes.get(infoset, [infoset])  # return a singleton containing the node of the same name by default


class SimpleExample(PerfectInfoExtensiveFormGame):

    players = ["Cora", "Prince"]
    
    _node_data = {
        "root": PerfectInfoNodeData(player="Prince", 
                                    actions=["pass","say 'swerve'"], 
                                    successor=["passed","spoke"]),
        "passed": PerfectInfoNodeData(player="Cora",
                                      actions=["pass","swerve"],
                                      successor=["passed,passed","passed,swerved"],
                                      event=["pass","swerve"]),
        "spoke": PerfectInfoNodeData(player="Cora",
                                      actions=["pass","swerve"],
                                      successor=["spoke,passed","spoke,swerved"],
                                      event=["pass","swerve"])
    }

    initial_node = "root"
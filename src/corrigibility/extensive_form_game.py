from dataclasses import dataclass
from collections import OrderedDict

@dataclass
class PerfectInfoNodeData:

#    id: str = None
#    """a short but unique id for this node that can serve as a key in a dictionary"""
    player: str = None
    """the player moving at this node"""
    actions: list = None
    """a list of actions available at this node"""
    successor_ids: list = None
    """a list of corresponding successor nodes"""
    events: list = None
    """an optional list of corresponding events (for agent players)"""
    probabilities: list = None
    """an optional list of probabilities (for probabilistic players)"""


@dataclass 
class ImperfectInfoNodeData(PerfectInfoNodeData):

    infoset_id: str = None
    """the id of the information set this node belongs to. If equal to id, this is a singleton infoset."""

@dataclass
class BeliefNodeData(ImperfectInfoNodeData):
    """having additional data on what the player believes to know about other players' policies"""

    policy_belief: dict = None
    """a dictionary, keyed by later node id, of lists representing the action probabilities for that later node believed by the player in the current node"""

class Choices(dict):
    """a dict, keyed by node_id or infoset_id, of actions"""
    pass

class Plan(Choices):
    """a Choices object that specifies an action_id for all infosets of one or more players"""
    pass


class _BaseGame:

    players: list = None
    """a list of the players in the game"""

    probabilistic_players: list = []
    """a list of players in the game who are modelled as following a known probabilistic policy (aka a behavior strategy)"""

    _node_data: dict = {}
    """a dict, keyed by node id, of node data objects"""

    def __init__(self, players=None, probabilistic_players=[], node_data=None):
        self.players = players
        self.probabilistic_players = probabilistic_players
        self.update_node_data(node_data)

    def update_node_data(self, dict):
        self._node_data.update(dict)


class PerfectInfoExtensiveFormGame(_BaseGame):

    initial_node_id: str = None
    """the id of the initial node at the start of the episode"""

    def __init__(self, initial_node_id="0", **kwargs):
        super().__init__(**kwargs)
        self.initial_node_id = initial_node_id

    def get_node_data(self, node_id) -> PerfectInfoNodeData:
        return self._node_data.get(node_id, PerfectInfoNodeData())  # return a terminal node by default


class ExtensiveFormGame(_BaseGame):

    # infoset_ids need to be distinct from node_ids!

    initial_infoset_id: str = None
    """the id of the initial infoset at the start of the episode"""
    initial_belief: dict = None
    """the initial belief about the initial state (dict of credence by node)"""

    _infosets: dict = {}
    """dict, keyed by infoset_id, of lists of nodes belonging to that infoset"""

    def __init__(self, initial_infoset_id="0", initial_belief={}, **kwargs):
        super().__init__(**kwargs)
        self.initial_infoset_id = initial_infoset_id
        self.initial_belief = initial_belief

    def get_node_data(self, node_id) -> ImperfectInfoNodeData:
        return self._node_data.get(node_id, ImperfectInfoNodeData())  # return a terminal node by default

    def get_nodes(self, infoset_id) -> list:
        """return the list of all node ids belonging to the same infoset"""
        return self._infosets.get(infoset_id, [infoset_id])  # return a singleton containing the node of the same name by default

    def update_node_data(self, dict):
        for nid, d in dict.items():
            try:
                self._infosets[self._node_data[nid].infoset_id].remove(nid)
            except:
                pass
            if d.successor_ids is None: d.successor_ids = []
            if d.infoset_id is None: d.infoset_id = nid
            self._infosets[d.infoset_id] = self._infosets.get(d.infoset_id, []) + [nid]
            self._node_data[nid] = d

#    def update_infosets(self, dict):
#        self._infosets.update(dict)

    def plans(self, player, clean=False): # -> Generator[Plan]:
        """Return a generator iterating through pairs of (plan, updated_ids) for the given player.
        A plan is a partial, deterministic policy specifying an action_id for each infoset that can be reached when all earlier actions conform to the same plan.
        The updated_ids entry is a list of infoset_ids of those infosets at which the plan has changed from the last yielded plan to the current yielded plan, or None if this is the first yielded plan.
        """
        # TODO!
        # Algorithm: 
        # - initialize by doing a forward pass, always taking the first action and pruning the branches of the other actions. keep the encountered nodes or infosets in a plan tree with nodes ordered by when they were encountered
        # - in each iteration: 
        #   - increment the action of the last infoset in that ordering, modulo number of actions, and mark that infoset as updated
        #   - if wrapped around: 
        #       - do the same for the infoset coming right before that infoset in the ordering.
        #       - if clean==True, delete the old action's branch from the plan tree
        #       - complete the plan tree by initializing the new action's branch as in the initialization step 
        #       - mark the infoset as updated
        #       - iterate by going back one step in the ordering until no longer wrapping around.
        plan = OrderedDict()
        # initialize the whole plan with action 0 in all encountered infosets:
        node_ids = self._infosets[self.initial_infoset_id]
        while len(node_ids) > 0:
            nid = node_ids.pop()
            d = self._node_data[nid]
            if d.player == player:
                iid = d.infoset_id
                plan[iid] = 0
                node_ids.append(d.successor_ids[0])
            else:
                node_ids.extend(d.successor_ids)
        yield plan, None
        # iterate:
        while True:
            updated_ids = []
            pos = len(plan)
            wrapped = True
            while wrapped:
                if pos == 0: return  # when the initial infoset's action was wrapped back to 0, we're finished 
                pos -= 1
                iid = list(plan.keys())[pos]
                i = self._infosets[iid]
                aid = plan[iid] 
                if clean:
                    # delete the branches we switched away from from the plan:
                    node_ids = [self._node_data[nid].successor_ids[aid] for nid in i]
                    while len(node_ids) > 0:
                        nid = node_ids.pop()
                        d = self._node_data[nid]
                        if d.player == player:
                            iid = d.infoset_id
                            node_ids.append(d.successor_ids[plan[iid]])
                            del plan[iid]
                        else:
                            node_ids.extend(d.successor_ids)
                aid += 1 
                if len(self._node_data[i[0]].actions) == aid:
                    # wrap back to action 0:
                    aid = 0
                else:
                    wrapped = False
                plan[iid] = aid
                updated_ids.append(iid)
                # initialize the branches we switched to with action 0:
                node_ids = [self._node_data[nid].successor_ids[aid] for nid in i]
                while len(node_ids) > 0:
                    nid = node_ids.pop()
                    d = self._node_data[nid]
                    if d.player == player:
                        iid = d.infoset_id
                        plan[iid] = 0
                        node_ids.append(d.successor_ids[0])
                    else:
                        node_ids.extend(d.successor_ids)
            yield plan, updated_ids


    def reduce(self, choices) -> _BaseGame:
        """return a reduced version of the game in which the infosets occurring in the given plan are removed and the links pointing to them are replaced by links to the respective action's successor nodes"""
        pass

class BeliefGame(ExtensiveFormGame):

    def get_node_data(self, node) -> BeliefNodeData:
        return self._node_data.get(node, BeliefNodeData())  # return a terminal node by default


class SimpleExample(PerfectInfoExtensiveFormGame):

    players = ["Cora", "Prince"]
    
    _node_data = {
        "root": PerfectInfoNodeData(player="Prince", 
                                    actions=["pass","say 'swerve'"], 
                                    successor_ids=["passed","spoke"]),
        "passed": PerfectInfoNodeData(player="Cora",
                                      actions=["pass","swerve"],
                                      successor_ids=["passed,passed","passed,swerved"],
                                      events=["pass","swerve"]),
        "spoke": PerfectInfoNodeData(player="Cora",
                                      actions=["pass","swerve"],
                                      successor_ids=["spoke,passed","spoke,swerved"],
                                      events=["pass","swerve"])
    }

    initial_node = "root"


if __name__ == "__main__":
    ND = ImperfectInfoNodeData
    Pas = ["foo","bar"]
    Aas = ["act","pass"]
    g = ExtensiveFormGame(players=["A","P","O","R"], probabilistic_players=["R"], initial_infoset_id="0", initial_belief={"0":1}, node_data={
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
"""Implements an AR-belief structure able to represent complex combinations of ambiguity and risk."""

import itertools
import numpy as np
from typing import Literal, Tuple, Dict, Set, List, Optional, NewType, Iterable, FrozenSet, Callable, Generator

NodeType = NewType("NodeType", Literal["risk", "ambiguity", "event"])
"""Possible types of nodes in an AR-belief structure
Risk nodes represent branchings with likelihoods attached.
Ambiguity nodes represent branchings with no likelihoods attached.
Event nodes are leaves representing events."""

Likelihood = NewType("Likelihood", float)

Label = NewType("Label", str)

Event = NewType("Event", FrozenSet)

RawData = NewType("RawData", 
                  Tuple[Literal["r"], Iterable[Tuple[Likelihood, "RawData"]] | Dict[Label, Tuple[Likelihood, "RawData"]]] 
                  | Tuple[Literal["a"], Iterable["RawData"] | Dict[Label, "RawData"]] 
                  | Tuple[Literal["e"], Iterable])
"""Raw data type for AR-belief structures."""

def indent_later(string: str, spaces: int = 2) -> str:
    """Indent a string by a given number of spaces from its 2nd line on."""
    return "\n".join([f"{' '*spaces}{line}" for line in string.split("\n")])[spaces:]

class ARBelief(object):
    """An AR-belief structure able to represent complex combinations of ambiguity and risk."""

    # Attributes:

    node_type: NodeType = None
    """The node type of the root node"""

    children: Optional[List] = None  # List[ARBelief]
    """The children of the root node (if node_type == "risk" or "ambiguity")"""

    labels: Optional[List[str | None]] = None
    """The optional labels of the links to the children nodes"""

    likelihoods: Optional[np.ndarray] = None
    """The likelihoods of the children nodes (if node_type == "risk")"""

    event: Optional[Event] = None
    """The event (= set of outcomes) represented by the root node (if node_type == "event")"""

    all_events: Set[Event] = None
    """The set of all events represented by the AR-belief structure"""

    _simplified: bool = False
    """Whether the AR-belief structure has been simplified"""

    _verified_disjoint_events: bool = False
    """Whether the events in the AR-belief structure have been verified to be disjoint"""

    # Initialization and representation:

    def __init__(self, 
                 data: Optional[RawData] = None, 
                 node_type: Optional[NodeType] = None, 
                 children: Optional[Iterable] = None, 
                 labels: Optional[Iterable[str]] = None, 
                 likelihoods: Optional[np.ndarray] = None, 
                 event: Optional[Event] = None, 
                 _verify_disjoint_events = True):
        """Initialize an AR-belief structure."""
        if data:
            assert node_type is None and children is None and likelihoods is None and event is None, "Cannot specify both data and node_type, children, likelihoods, or event"
            # assert isinstance(data, RawData)
            self.node_type = {"r":"risk", "a":"ambiguity", "e":"event"}[data[0]]
            if self.node_type == "risk":
                if isinstance(data[1], dict):
                    self.labels = list(data[1].keys())
                    self.likelihoods = np.array([t[0] for t in data[1].values()])
                    self.children = [ARBelief(data=t[1], _verify_disjoint_events=False) for t in data[1].values()]
                else:
                    self.labels = [None] * len(data[1])
                    self.likelihoods = np.array([t[0] for t in data[1]])
                    self.children = [ARBelief(data=t[1], _verify_disjoint_events=False) for t in data[1]]
            elif self.node_type == "ambiguity":
                children_data = None
                if isinstance(data[1], dict):
                    children_data = list(data[1].values())
                    self.labels = list(data[1].keys())
                else:
                    children_data = data[1]
                    self.labels = [None] * len(children_data)
                self.children = [ARBelief(data=t, _verify_disjoint_events=False) for t in children_data]
            elif self.node_type == "event":
                self.event = frozenset(data[1])
        else:
            assert node_type is not None, "Must specify either data or node_type"
            self.node_type = node_type
            if self.node_type == "risk":
                self.children = list(children)
                self.labels = list(labels) if labels else [None] * len(children)
                self.likelihoods = np.array(likelihoods)
            elif self.node_type == "ambiguity":
                self.children = children
                self.labels = labels if labels else [None] * len(children)
            elif self.node_type == "event":
                self.event = frozenset(event)
        if self.node_type == "risk":
            assert np.all(self.likelihoods >= 0) and np.sum(self.likelihoods) <= 1, "Likelihoods must be non-negative and sum to at most unity"
        self.all_events = set()
        if self.event: 
            self.all_events.add(self.event)
        if self.children:
            for c in self.children:
                self.all_events.update(c.all_events)
        # self-simplify:
        simplified = self._simplify()
        self.node_type = simplified.node_type
        self.children = simplified.children
        self.labels = simplified.labels
        self.likelihoods = simplified.likelihoods
        self.event = simplified.event
        self.all_events = simplified.all_events
        self._simplified = True
        if _verify_disjoint_events:
            es = list(self.all_events)
            for i, e1 in enumerate(es):
                for e2 in es[i+1:]:
                    assert e1 == e2 or e1.isdisjoint(e2), f"Events {set(e1)} and {set(e2)} are not disjoint"
            self._verified_disjoint_events = True

    def __str__(self) -> str:
        """Return a string representation of the AR-belief structure."""
        if self.node_type == "risk":
            return f"Risk [\n  {",\n  ".join([f"{p} {l+' ' if l else ''}{indent_later(str(b))}" for (p,l,b) in zip(self.likelihoods, self.labels, self.children)])}  ]"
        elif self.node_type == "ambiguity":
            return f"Ambiguity (\n  {",\n  ".join([f"{l+' ' if l else ''}{indent_later(str(b))}" for (b,l) in zip(self.children, self.labels)])}  )"
        elif self.node_type == "event":
            return f"Event {str(set(self.event))}"
        
    def save_figure(self, filename: str, format: str = "pdf"):
        """Save a visual representation of the AR-belief structure to a file, laid out using graphviz."""
        raise NotImplementedError


    # Simplification:

    def _simplify(self) -> "ARBelief":
        """Simplify the AR-belief structure by:
        - removing all ambiguity nodes with only one, unlabelled child
        - removing all risk nodes with only one, unlabelled child and likelihood 1
        - collapsing consecutive unlabelled ambiguity nodes
        - collapsing consecutive unlabelled risk nodes
        - collapsing equal children of risk nodes and summing their likelihoods
        - removing duplicate children of ambiguity nodes"""
        if self._simplified:
            return self
        belief = self
        change = True
        while change:
            belief, change = belief._remove_ambiguity_w_unlabelled_single_child(change)
            belief, change = belief._remove_risk_w_single_unlabelled_certain_child(change)
            belief, change = belief._collapse_consecutive_unlabelled_risks(change)
            belief, change = belief._collapse_consecutive_unlabelled_ambiguities(change)
            belief, change = belief._collapse_equal_children_of_risk(change)
            belief, change = belief._remove_duplicate_children_of_ambiguity(change)
        return belief

    def _remove_ambiguity_w_unlabelled_single_child(self, change: bool) -> Tuple["ARBelief", bool]:
        """Remove an ambiguity node with only one child, if any."""
        if self.node_type == "event":
            return self, False
        # check if the current node is an ambiguity node with only one child:
        elif self.node_type == "ambiguity" and len(self.children) == 1 and self.labels[0] is None:
            return self.children[0], True
        else:
            # check if any of the children is an ambiguity node with only one child:
            children, changes = zip(*[child._remove_ambiguity_w_unlabelled_single_child(change) for child in self.children])
            if any(changes):
                return ARBelief(node_type=self.node_type, children=children, labels=self.labels, likelihoods=self.likelihoods, _verify_disjoint_events=False), True
            else:
                return self, False

    def _remove_risk_w_single_unlabelled_certain_child(self, change: bool) -> Tuple["ARBelief", bool]:
        """Remove a risk node with only one child and likelihood 1, if any."""
        if self.node_type == "event":
            return self, False
        # check if the current node is a risk node with only one child and likelihood 1:
        elif self.node_type == "risk" and len(self.children) == 1 and self.likelihoods[0] == 1 and self.labels[0] is None:
            return self.children[0], True
        else:
            # check if any of the children is a risk node with only one child and likelihood 1:
            children, changes = zip(*[child._remove_risk_w_single_unlabelled_certain_child(change) for child in self.children])
            if any(changes):
                return ARBelief(node_type=self.node_type, children=children, labels=self.labels, likelihoods=self.likelihoods, _verify_disjoint_events=False), True
            else:
                return self, False
            
    def _root_links_are_unlabelled(self) -> bool:
        """Check if the current node is unlabelled."""
        return self.labels is None or all([l is None for l in self.labels])
    
    def _collapse_consecutive_unlabelled_risks(self, change: bool) -> Tuple["ARBelief", bool]:
        """Collapse consecutive risk nodes, if any."""
        if self.node_type == "risk" and self._root_links_are_unlabelled():
            # find the first child that is also a risk node:
            i = next((i for i, child in enumerate(self.children) if child.node_type == "risk" and child._root_links_are_unlabelled()), None)
            if i is not None:
                # in this node's children, replace the child by its children and multiply the likelihoods:
                children = self.children[:i] + self.children[i].children + self.children[i+1:]
                labels = [None] * len(children)
                likelihoods = np.concatenate([self.likelihoods[:i], self.likelihoods[i] * self.children[i].likelihoods, self.likelihoods[i+1:]])
                return ARBelief(node_type="risk", children=children, labels=labels, likelihoods=likelihoods, _verify_disjoint_events=False), True
        return self, False
    
    def _collapse_consecutive_unlabelled_ambiguities(self, change: bool) -> Tuple["ARBelief", bool]:
        """Collapse consecutive ambiguity nodes, if any."""
        if self.node_type == "ambiguity" and self._root_links_are_unlabelled():
            # find the first child that is also an ambiguity node:
            i = next((i for i, child in enumerate(self.children) if child.node_type == "ambiguity" and child._root_links_are_unlabelled()), None)
            if i is not None:
                # in this node's children, replace the child by its children:
                children = self.children[:i] + self.children[i].children + self.children[i+1:]
                labels = [None] * len(children)
                return ARBelief(node_type="ambiguity", children=children, labels=labels, _verify_disjoint_events=False), True
        return self, False

    def _collapse_equal_children_of_risk(self, change: bool) -> Tuple["ARBelief", bool]:
        """Collapse equal children of risk nodes and sum their likelihoods, if any."""
        if self.node_type == "risk":
            # find the first pair of equal children:
            for i1, c1 in enumerate(self.children):
                for i2, c2 in enumerate(self.children[i1+1:], start=i1+1):
                    if c1 == c2 and self.labels[i1] == self.labels[i2]:
                        # remove c2 and sum the likelihoods:
                        children = self.children[:i2] + self.children[i2+1:]
                        labels = self.labels[:i2] + self.labels[i2+1:]
                        likelihoods = np.concatenate([self.likelihoods[:i1], [self.likelihoods[i1] + self.likelihoods[i2]], self.likelihoods[i1+1:i2], self.likelihoods[i2+1:]])
                        return ARBelief(node_type="risk", children=children, labels=labels, likelihoods=likelihoods, _verify_disjoint_events=False), True
        return self, False
    
    def _remove_duplicate_children_of_ambiguity(self, change: bool) -> Tuple["ARBelief", bool]:
        """Remove duplicate children of ambiguity nodes, if any."""
        if self.node_type == "ambiguity":
            # find the first pair of equal children:
            for i1, c1 in enumerate(self.children):
                for i2, c2 in enumerate(self.children[i1+1:], start=i1+1):
                    if c1 == c2 and self.labels[i1] == self.labels[i2]:
                        # remove c2:
                        children = self.children[:i2] + self.children[i2+1:]
                        labels = self.labels[:i2] + self.labels[i2+1:]
                        return ARBelief(node_type="ambiguity", children=children, labels=labels, _verify_disjoint_events=False), True
        return self, False


    # Standard operators:

    def __eq__(self, other: "ARBelief") -> bool:
        """Check if two AR-belief structures are equal (ignoring ordering of children)."""
        self = self._simplify()
        other = other._simplify()
        if self.node_type != other.node_type:
            return False
        if self.node_type == "risk":
            if not np.array_equal(np.sort(self.likelihoods), np.sort(other.likelihoods)):
                return False
            for i, child in enumerate(self.children):
                j = next((j for j, other_child in enumerate(other.children) if child == other_child), None)
                if j is None:
                    return False
                if self.likelihoods[i] != other.likelihoods[j]:
                    return False
                if self.labels[i] != other.labels[j]:
                    return False
        elif self.node_type == "ambiguity":
            if len(self.children) != len(other.children):
                return False
            for i, child in enumerate(self.children):
                j = next((j for j, other_child in enumerate(other.children) if child == other_child), None)
                if j is None:
                    return False
                if self.labels[i] is not None and other.labels[i] is not None and self.labels[i] != other.labels[j]:
                    return False
        elif self.node_type == "event":
            if self.event != other.event:
                return False
        return True
    
    def __ne__(self, other: "ARBelief") -> bool:
        """Check if two AR-belief structures are not equal (ignoring ordering of children)."""
        return not self.__eq__(other)


    # Operations:

    def conditioned_on_event(self, condition: Event) -> "ARBelief":
        """Condition the AR-belief structure on an event.
        Note that likelihoods will not be renormalized."""
        if self.node_type == "event":
            if self.event.issubset(condition):
                return ARBelief(node_type="event", event=self.event)
            elif self.event.isdisjoint(condition):
                return None
            else:
                raise ValueError("Conditioning on a condition event that is incompatible with this belief's events")
        else:
            children0 = [child.conditioned_on_event(condition) for child in self.children]
            children = []
            labels = []
            likelihoods = [] if self.node_type == "risk" else None
            for i, child in enumerate(children0):
                if child is not None:
                    children.append(child)
                    labels.append(self.labels[i])
                    if self.node_type == "risk":
                        likelihoods.append(self.likelihoods[i])
            return ARBelief(node_type=self.node_type, children=children, labels=labels, likelihoods=likelihoods)

    def restricted_to_labels(self, labels: Iterable[Label]) -> "ARBelief":
        """Remove all branches starting with a non-None label not in labels."""
        if self.node_type == "event":
            return self
        else:
            children0 = [child.restricted_to_labels(labels) for child in self.children]
            children = []
            ls = []
            likelihoods = [] if self.node_type == "risk" else None
            for i, child in enumerate(children0):
                if child is not None and (self.labels[i] is None or self.labels[i] in labels):
                    children.append(child)
                    ls.append(self.labels[i])
                    if self.node_type == "risk":
                        likelihoods.append(self.likelihoods[i])
            res = ARBelief(node_type=self.node_type, children=children, labels=ls, likelihoods=likelihoods, _verify_disjoint_events=False) if children else None
            #print(labels, self.labels, res)
            return res

    def coarsened_to(self, coarsening: Iterable[Event]) -> "ARBelief":
        """Coarsen the AR-belief structure according to a coarsening of events."""
        if self.node_type == "event":
            # find the first event in the coarsening that contains this event:
            i = next((i for i, c in enumerate(coarsening) if self.event.issubset(c)), None)
            assert i is not None, "Coarsening to an event system that is not compatible with this belief's events"
            return ARBelief(node_type="event", event=coarsening[i])
        else:
            children = [child.coarsened_to(coarsening) for child in self.children]
            return ARBelief(node_type=self.node_type, children=children, labels=self.labels, likelihoods=self.likelihoods)
    
    def unlabelled(self, keep: Optional[Iterable[Label]] = None) -> "ARBelief":
        """Remove all labels except those named in keep from the AR-belief structure."""
        labels = [l if keep is not None and l in keep else None for l in self.labels] if self.labels else None
        if self.node_type == "event":
            return self
        else:
            return ARBelief(node_type=self.node_type, children=[child.unlabelled(keep=keep) for child in self.children], labels=labels, likelihoods=self.likelihoods, _verify_disjoint_events=False)
        
    def extended_by(self, events: Iterable[Event], beliefs: Iterable["ARBelief"]) -> "ARBelief":
        """Extend the AR-belief structure by replacing certain events by certain beliefs."""
        assert len(events) == len(beliefs)
        if self.node_type == "event":
            i = next((i for i, e in enumerate(events) if self.event == e), None)
            return beliefs[i] if i is not None else self
        else:
            children = [child.extended_by(events, beliefs) for child in self.children]
            return ARBelief(node_type=self.node_type, children=children, labels=self.labels, likelihoods=self.likelihoods, _verify_disjoint_events=False)


    # Scenarios:

    def scenarios(self) -> Generator:
        """Generate all scenarios represented by the AR-belief structure.
        A scenario is the result of restricting the tree to one outgoing link for each ambiguity node."""
        if self.node_type == "ambiguity":
            for child in self.children:
                for scenario in child.scenarios():
                    yield ARBelief(node_type="ambiguity", children=[scenario], labels=[self.labels[0]], _verify_disjoint_events=False)
        elif self.node_type == "risk":
            # loop through all combinations of scenarios, one for each child, using itertools.product:
            for scenario_combination in itertools.product(*[child.scenarios() for child in self.children]):
                yield ARBelief(node_type="risk", children=list(scenario_combination), labels=self.labels, likelihoods=self.likelihoods, _verify_disjoint_events=False)
        else:
            yield self

    def scenario_weighted_total(self, func: Callable) -> float:
        """Compute the weighted total of event func in this given scenario, using event's likelihoods as weights."""
        if self.node_type == "event":
            return func(self.event)
        elif self.node_type == "risk":
            return np.dot(self.likelihoods, [child.scenario_weighted_total(func) for child in self.children])
        else:
            raise ValueError("Cannot compute expectation in an AR-belief structure with ambiguity nodes")

    def scenario_total_likelihood(self) -> float:
        """Compute the sum of all leaves' likelihoods of an AR-belief structure without ambiguity nodes."""
        return self.scenario_weighted_total(lambda e: 1)

    def scenario_expectation(self, func: Callable) -> float:
        """Compute the expectation of event func in this AR-belief structure."""
        return self.scenario_weighted_total(func) / self.scenario_total_likelihood()


    # Label-defined partial scenarios:

    def conditioned_on_label(self, label: Label) -> "ARBelief":
        """Condition the AR-belief structure on the occurrence of the label on the path."""
        belief, contains_label = self._conditioned_on_label_or_self(label)
        if contains_label:
            return belief
        else:
            raise ValueError(f"Label {label} not found in the AR-belief structure")

    def _conditioned_on_label_or_self(self, label: Label) -> Tuple["ARBelief", bool]:
        if self.node_type == "event":
            return self, False
        else:
            children, contain_label = zip(*[child._conditioned_on_label_or_self(label) for child in self.children])
            indices = [i for i in range(len(children)) if self.labels[i] == label or contain_label[i]]
            if len(indices) == 0:
                return self, False
            else:
                # return only those branches that contain the label:
                return ARBelief(node_type=self.node_type, children=[children[i] for i in indices], labels=[self.labels[i] for i in indices], likelihoods=[self.likelihoods[i] for i in indices] if self.likelihoods is not None else None, _verify_disjoint_events=False), True


    # Evaluation:

    def plausibility_weighted_min_expectation(self, func: Callable) -> float:
        """Compute the plausibility-weighted worst-case expectation of given event func given the AR-belief structure."""
        plausibilities_and_values = [(sc.scenario_total_likelihood(), sc.scenario_expectation(func)) for sc in self.scenarios()]
        # sort by decreasing plausibility:
        plausibilities_and_values.sort(key=lambda x: -x[0])
        plausibilities, values = zip(*plausibilities_and_values)
        # compute the list of plausibility decrements:
        plausibility_decrements = [plausibilities[i] - plausibilities[i+1] for i in range(len(plausibilities)-1)] + [plausibilities[-1]]
        return np.dot(plausibility_decrements, [min(values[:i]) for i in range(1, len(plausibilities_and_values)+1)])
    
    def evaluate_action(self, action: Label, valuation: Callable) -> float:
        """Evaluate an action according to the given event valuation given the AR-belief structure."""
        partial_scenario = self.conditioned_on_label(action).unlabelled()
        return partial_scenario.plausibility_weighted_min_expectation(valuation)


if __name__ == "__main__":
    # Example usage
    data = ("r", [
                    (0.3, ("a", [
                                ("r", {
                                        "up": (0.5, ("e", [1,2,3])), 
                                        "down": (0.5, ("e", [4,5,6]))
                                    }),
                                ("e", [7])
                            ])), 
                    (0.7, ("e", [8,9]))
                ])
    arb = ARBelief(data)
    print(arb)
    data2 = ("a", [
                ("r", {
                    "up": (0.7, ("e", [9,8])),
                    "down": (0.3, ("a", {
                                "left": ("e", [7]),
                                "right": ("e", [7]),
                                None: ("r", {
                                        None: (0.5, ("e", [1,2,3])), 
                                        "what": (0.5, ("e", [4,6,5]))
                                    })
                                }))
                })
            ])
    arb2 = ARBelief(data2)
    print(arb2)
    print(arb == arb2)
    arb3 = arb2.conditioned_on_event({1,2,3,7})
    print(arb3)
    arb4 = arb2.coarsened_to([{1,2,3,4,5,6,7,8,9}])
    print(arb4)
    arb5 = arb2.unlabelled(keep=["what"]).coarsened_to([{1,2,3,4,5,6},{7,8,9}])
    print(arb5)
    subs1 = ARBelief(("e", [10,20,30]))
    subs2 = ARBelief(("r", [(0.1, ("e", [10,20,30])), (0.2, ("e", [40,50,60]))]))
    arb6 = arb5.extended_by([{1,2,3,4,5,6},{7,8,9}], [subs1, subs2])
    print(arb6)
    arb7 = arb2.restricted_to_labels(["up", "down", "what"])
    print(arb7)
    print()
    print(arb2)
    print("Scenarios:")
    for arb in arb2.unlabelled().scenarios():
        print(arb)

    print()
    data = ("a", [
                ("r", [
                    (0.3, ("a", {
                    "left": ("a", {
                                "up": ("e", [0]), 
                                "down": ("e", [1])  }), 
                    "right": ("r", {
                                "up": (0.5, ("e", [0])), 
                                "down": (0.5, ("e", [1]))  })
                        })),
                    (0.7, ("a", {
                    "right": ("a", {
                                "up": ("e", [0]), 
                                "down": ("e", [1])  }), 
                    "left": ("r", {
                                "up": (0.5, ("e", [0])), 
                                "down": (0.5, ("e", [1]))  })  }))  ]),
                ("r", [
                    (0.5, ("a", {
                    "left": ("a", {
                                "up": ("e", [0]), 
                                "down": ("e", [1])  }), 
                    "right": ("r", {
                                "up": (0.5, ("e", [0])), 
                                "down": (0.5, ("e", [1]))  })
                        })),
                    (0.2, ("a", {
                    "right": ("a", {
                                "up": ("e", [0]), 
                                "down": ("e", [1])  }), 
                    "left": ("r", {
                                "up": (0.5, ("e", [0])), 
                                "down": (0.5, ("e", [1]))  })  }))  ])  ])
            
    arb = ARBelief(data)
    print()
    print(arb)
    for action in ["left", "right"]:
        print(f"Action {action} has evaluation {arb.evaluate_action(action, lambda e: list(e)[0])}")

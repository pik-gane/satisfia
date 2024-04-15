import pytest
from world_model.objects import Agent, Delta, DeltaState, EmptyToWall, EmptyToWallState
from world_model.simple_gridworld import (
    Box,
    GlassState,
    SimpleGridworld,
    Direction,
    Glass,
)


def print_td(td: dict):
    print("==============")
    for gw, (prob, exact) in td.items():
        print(f"- prob {prob} ({'non-' if not exact else ''}exact), t={gw.t}")
        print(gw)


def get_where_agent_at(
    td, loc: tuple[int, int]
) -> tuple[SimpleGridworld, float] | None:
    for gw, prob in td.items():
        if gw.agents[0].location == loc:
            return gw, prob[0]


def test_pinnacle_one_way():
    g: SimpleGridworld = SimpleGridworld(
        [
            ["X", "A", "^"],
            ["~", "X", " "],
            ["X", " ", " "],
        ],
    )
    a = g.agents[0]
    assert g._can_move(Direction.DOWN, a)
    assert g._can_move(Direction.RIGHT, a)
    assert not g._can_move(Direction.LEFT, a)
    assert not g._can_move(Direction.UP, a)
    td = g.transition_distribution(a, Direction.RIGHT)
    assert len(td) == 1
    [(gw, (prob, _))] = td.items()
    assert gw.agents[0].location == (2, 1)
    assert prob == 1

    td = g.transition_distribution(a, Direction.RIGHT)


def test_pinnacle_threeway():
    g: SimpleGridworld = SimpleGridworld(
        [
            [" ", " ", " "],
            ["A", "^", " "],
            ["X", "X", " "],
            ["X", " ", " "],
        ],
    )
    a = g.agents[0]
    td = g.transition_distribution(a, Direction.RIGHT)
    print_td(g.transition_distribution(a, Direction.RIGHT))

    assert len(td) == 3
    for loc in ((1, 0), (2, 1), (1, 2)):
        w, prob = get_where_agent_at(td, loc)
        assert w.t == g.t + 1
        assert prob == 1.0 / 3
        box_moved = w.at((1, 3)).get(Box)
        assert bool(box_moved) == (loc == (1, 2))


def test_unstable():
    g: SimpleGridworld = SimpleGridworld(
        [
            [" ", " ", " "],
            ["A", "~", " "],
            ["X", "X", " "],
            ["X", " ", " "],
        ],
        uneven_ground_prob=0.6,
    )
    a = g.agents[0]
    td = g.transition_distribution(a, Direction.RIGHT)
    print_td(g.transition_distribution(a, Direction.RIGHT))

    assert len(td) == 4
    for loc in ((1, 0), (2, 1), (1, 2)):
        w, prob = get_where_agent_at(td, loc)
        assert w.t == g.t + 1
        assert prob == (1.0 - 0.4) / 3
        box_moved = w.at((1, 3)).get(Box)
        assert bool(box_moved) == (loc == (1, 2))

    w, prob = get_where_agent_at(td, (1, 1))
    assert prob == 0.4


def test_glass():
    g: SimpleGridworld = SimpleGridworld(
        [
            ["#", " ", "#"],
            ["#", "|", "#"],
            ["|", "A", "|"],
            ["X", "|", " "],
            ["X", "X", " "],
        ],
        uneven_ground_prob=0.6,
    )
    a = g.agents[0]

    td = g.transition_distribution(a, Direction.RIGHT)
    assert len(td) == 1
    w, prob = get_where_agent_at(td, (2, 2))
    assert w.t == g.t + 1
    assert prob == 1
    glass = w.at((2, 2)).get(Glass)
    assert glass.state is GlassState.BROKEN

    td = g.transition_distribution(a, Direction.UP)
    assert len(td) == 1
    w, prob = get_where_agent_at(td, (1, 1))
    assert prob == 1
    glass = w.at((1, 0)).get(Glass)
    assert glass.state is None

    with pytest.raises(AssertionError):
        td = g.transition_distribution(a, Direction.DOWN)

    td = g.transition_distribution(a, Direction.LEFT)
    assert len(td) == 1
    w, prob = get_where_agent_at(td, (0, 2))
    assert prob == 1
    glass = w.at((0, 2)).get(Glass)
    assert glass.state is GlassState.BROKEN


def test_btw():
    g: SimpleGridworld = SimpleGridworld(
        [
            ["#", " ", "#"],
            ["#", ",", "#"],
            ["|", "A", "|"],
        ],
    )
    a = g.agents[0]
    etw = g.at((1, 1)).get(EmptyToWall)
    assert etw.state is None

    td = g.transition_distribution(a, Direction.UP)
    assert len(td) == 1
    w1, prob = get_where_agent_at(td, (1, 1))
    assert w1.t == g.t + 1
    assert prob == 1
    etw = w1.at((1, 1)).get(EmptyToWall)
    assert etw.state is None

    td = w1.transition_distribution(w1.agents[0], Direction.UP)
    assert len(td) == 1
    w2, prob = get_where_agent_at(td, (1, 0))
    assert w2.t == w1.t + 1
    assert prob == 1
    etw = w2.at((1, 1)).get(EmptyToWall)
    assert etw.state is EmptyToWallState.WALL

    with pytest.raises(AssertionError):
        td = w2.transition_distribution(w2.agents[0], Direction.UP)
    with pytest.raises(AssertionError):
        td = w2.transition_distribution(w2.agents[0], Direction.DOWN)


def test_delta():
    g: SimpleGridworld = SimpleGridworld(
        [
            ["A", "Δ", " "],
        ],
        [
            [" ", "D", " "],
        ],
        cell_code2delta = {"D": 3}
    )
    delta = g.at((1, 0)).get(Delta)
    assert delta.state is None
    assert g.agents[0].collected_delta == 0

    td = g.transition_distribution(g.agents[0], Direction.RIGHT)
    assert len(td) == 1
    w1, prob = get_where_agent_at(td, (1, 0))
    assert w1.t == g.t + 1
    assert prob == 1

    td = w1.transition_distribution(w1.agents[0], Direction.RIGHT)
    assert len(td) == 1
    w2, prob = get_where_agent_at(td, (2, 0))
    assert w2.t == w1.t + 1
    assert prob == 1
    delta = w2.at((1, 0)).get(Delta)
    assert delta.state is DeltaState.COLLECTED
    assert w2.agents[0].collected_delta == 3

    td = w2.transition_distribution(w2.agents[0], Direction.LEFT)
    assert len(td) == 1
    w3, prob = get_where_agent_at(td, (1, 0))
    print(w3)
    print(w3.at((1,0)).get(Agent).collected_delta)
    assert w3.t == w2.t + 1
    assert prob == 1
    assert w3.agents[0].collected_delta == 3

    td = w3.transition_distribution(w3.agents[0], Direction.LEFT)
    assert len(td) == 1
    w4, prob = get_where_agent_at(td, (0, 0))
    assert w4.t == w3.t + 1
    assert prob == 1
    delta = w4.at((1, 0)).get(Delta)
    assert delta.state is DeltaState.COLLECTED
    assert w4.agents[0].collected_delta == 3

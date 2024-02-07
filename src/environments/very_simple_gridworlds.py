from world_model import SimpleGridworld

def make_simple_gridworld(gw="GW1", time=None, **kwargs):

    if gw == "GW1":
        grid = [
          ['#', '#', '#', '#', '#'],
          ['#', '#', 'G' ,'#', '#'],
          ['#', 'G', 'A', 'G', '#'],
          ['#', '#', '#', '#', '#']
        ]
        delta_grid = [
          [' ', ' ', ' ', ' ', ' '],
          [' ', ' ','G2' ,' ', ' '],
          [' ','G1', ' ','G3', ' '],
          [' ', ' ', ' ', ' ', ' ']
        ]
        expectedDeltaTable = { 'G1': 1, 'G2': 2, 'G3': 3 }
        aleph0 = [1.9, 2.1]
        totalTime = time or 10

    elif gw == "GW2":
        grid = [
          ['#', '#', '#', '#', '#'],
          ['#', 'A', ' ', ' ', '#'],
          ['#', 'G', 'G' ,'G', '#'],
          ['#', '#', '#', '#', '#']
        ],
        delta_grid = [
          [' ', ' ', ' ', ' ', ' '],
          [' ', ' ', ' ', ' ', ' '],
          [' ','Ga','Gb','Gc', ' '],
          [' ', ' ', ' ', ' ', ' ']
        ],
        expectedDeltaTable = { 'Ga': 1, 'Gb': 3, 'Gc': 2 }
        aleph0 =  [1.9, 2.1]
        totalTime = time or 10

    elif gw == "GW3":
        grid = [
          ['#', '#', '#', '#', '#'],
          ['#', 'G', '#', 'G', '#'],
          ['#', ' ', 'A', ' ', '#'],
          ['#', 'G', '#', 'G', '#'],
          ['#', '#', '#', '#', '#']
        ]
        delta_grid = [
          [' ', ' ', ' ', ' ', ' '],
          [' ','Ga', ' ','Gc', ' '],
          [' ', ' ', ' ', ' ', ' '],
          [' ','Gb', ' ','Gd', ' '],
          [' ', ' ', ' ', ' ', ' ']
        ]
        expectedDeltaTable = { 'Ga': 0, 'Gb': 2, 'Gc': 1, 'Gd': 3 }
        aleph0 = [1.9, 2.1]
        totalTime = time or 10

    elif gw == "GW4":
        grid = [
          ['#', '#', '#', '#', '#', '#'],
          ['#', 'A', ' ', ' ', ' ', '#'],
          ['#', 'G', 'G' ,'G', 'G', '#'],
          ['#', '#', '#', '#', '#', '#']
        ]
        delta_grid = [
          [' ', ' ', ' ', ' ', ' '],
          [' ', ' ', ' ', ' ', ' '],
          [' ','Ga','Gb','Gc','Gd'],
          [' ', ' ', ' ', ' ', ' ']
        ]
        expectedDeltaTable = { 'Ga': 1, 'Gb': 3, 'Gc': 2, 'Gd': 0 }
        aleph0 = [1.4, 1.6]
        totalTime = time or 10

    elif gw == "GW5":
        grid = [
          ['#', '#', '#', '#', '#'],
          ['#', 'G', '#', 'G', '#'],
          ['#', ' ', 'A', '^', '#'],
          ['#', 'G', '#', 'G', '#'],
          ['#', '#', '#', '#', '#']
        ]
        delta_grid = [
          [' ', ' ', ' ', ' ', ' '],
          [' ','G1', ' ','G1', ' '],
          [' ', ' ', ' ', ' ', ' '],
          [' ','G3', ' ','G3', ' '],
          [' ', ' ', ' ', ' ', ' ']
        ]
        expectedDeltaTable = { 'G1': 1, 'G3': 3 }
        aleph0 = [1.9, 2.1]
        totalTime = time or 10

    elif gw == "GW6":
        grid = [
          ['#', '#', '#', '#', '#'],
          ['#', 'G', '#', 'G', '#'],
          ['#', ' ', 'A', ' ', '#'],
          ['#', 'G', '#', '#', '#'],
          ['#', '#', '#', '#', '#']
        ]
        delta_grid = [
          [' ', ' ', ' ', ' ', ' '],
          [' ','G1', ' ','G4', ' '],
          [' ', ' ', ' ', 'Δ', ' '],
          [' ','G3', ' ', ' ', ' '],
          [' ', ' ', ' ', ' ', ' ']
        ]
        expectedDeltaTable = { 'G1': 1, 'G3': 3, 'G4': 4, 'Δ': -2 }
        aleph0 = 2
        totalTime = time or 10

    elif gw == "test_box":
        grid = [
            [' ', 'X', ' ', 'X', 'A', 'X', 'G', ' ', ' ']
        ]  
        delta_grid = [
            [' ', ' ', ' ', ' ', ' ', ' ', 'G', ' ', ' ']
        ]
        expectedDeltaTable = { 'G': 1 }
        aleph0 = 1
        totalTime = time or 20

    elif gw == "AISG2":
        grid = [
          ['#', '#', '#', '#', '#', '#'],
          ['#', ' ', 'A', '#', '#', '#'],
          ['#', ' ', 'X', ' ', ' ', '#'],
          ['#', '#', ' ', ' ', ' ', '#'],
          ['#', '#', '#', ' ', 'G', '#'],
          ['#', '#', '#', '#', '#', '#']
        ]
        delta_grid = [
          [' ', ' ', ' ', ' ', ' ', ' '],
          [' ', ' ', ' ', ' ', ' ', ' '],
          [' ', ' ', ' ', ' ', ' ', ' '],
          [' ', ' ', ' ', ' ', ' ', ' '],
          [' ', ' ', ' ', ' ', 'G', ' '],
          [' ', ' ', ' ', ' ', ' ', ' ']
        ]
        expectedDeltaTable = { 'G': 1 }
        aleph0 = 1
        totalTime = time or 15

    return (SimpleGridworld(grid=grid, delta_grid=delta_grid,
                            cell_code2delta=expectedDeltaTable, max_episode_length=totalTime,
                            **kwargs), 
            aleph0)

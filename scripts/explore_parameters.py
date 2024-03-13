import sys
sys.path.insert(0,'./src/')

import time
import PySimpleGUI as sg

from environments.very_simple_gridworlds import make_simple_gridworld
from satisfia.agents.makeMDPAgentSatisfia import AgentMDPPlanning

gridworlds = ["AISG2", "GW1", "GW2", "GW3", "GW4", "GW5", "GW6", "GW22", "test_box"]
parameter_data = [
    ("lossTemperature", 0, 10, 0.1),
    ("lossCoeff4Variance", 0, 100, 0),
    ('lossCoeff4Random', 0, 100, 0), 
    ('lossCoeff4FeasibilityPower', 0, 100, 0), 
    ('lossCoeff4LRA1', 0, 100, 0), 
    ('lossCoeff4Time1', 0, 100, 0), 
    ('lossCoeff4Entropy1', 0, 100, 0), 
    ('lossCoeff4KLdiv1', 0, 100, 0), 
    ('lossCoeff4DP', 0, 100, 0), 
    ('lossCoeff4Fourth', 0, 100, 0), 
    ('lossCoeff4Cup', 0, 100, 0), 
    ('lossCoeff4LRA', 0, 100, 0), 
    ('lossCoeff4Time', 0, 100, 0), 
    ('lossCoeff4DeltaVariation', 0, 100, 0), 
    ('lossCoeff4Entropy', 0, 100, 0), 
    ('lossCoeff4KLdiv', 0, 100, 0),
    ('lossCoeff4TrajectoryEntropy', 0, 100, 0), 
    ('minLambda', 0, 1, 0), 
    ('maxLambda', 0, 1, 1), 
] # name, min, max, initial

# Create a drop down for selecting the gridworld
gridworld_dropdown = sg.DropDown(gridworlds, default_value=gridworlds[0], key='gridworld_dropdown')

# Create sliders for setting the parametersers
parameter_sliders = {}
for pd in parameter_data:
    slider = sg.Slider(range=(pd[1], pd[2]), default_value=pd[3], orientation='h', key=pd[0])
    parameter_sliders[pd[0]] = slider


# Create buttons for starting, pausing, stepping, and continuing the simulation
restart_button = sg.Button("(Re)start", key='restart_button')
pause_button = sg.Button("Pause", key='pause_button')
step_button = sg.Button("Step", key='step_button')
continue_button = sg.Button("Continue", key='continue_button')
speed_slider = sg.Slider(range=(1, 20), default_value=10, orientation='h', key='speed_slider')

# Create the layout
layout = [
    [sg.Text("Gridworld"), gridworld_dropdown],
    [sg.Column([[sg.Text(pd[0]), parameter_sliders[pd[0]]] for pd in parameter_data], element_justification='r')],
    [restart_button, pause_button, step_button, continue_button], 
    [sg.Text("Speed"), speed_slider]
]

# Create the window

window = sg.Window("Explore Parameters", layout)

gridworld = gridworlds[0]
parameter_values = { pd[0]: pd[3] for pd in parameter_data }
env = None
agent = None
running = False
stepping = False
terminated = False

while True:
    parsed_events = False
    while not parsed_events:
        event, values = window.read(timeout=0)
        if event == sg.WINDOW_CLOSED:
            break
        elif event == 'restart_button':
            gridworld = values['gridworld_dropdown']
            parameter_values = { pd[0]: values[pd[0]] for pd in parameter_data }
            print("RESTART gridworld", gridworld, parameter_values)
            env, aleph = make_simple_gridworld(gw=gridworld, render_mode="human", fps=values['speed_slider'])
            state, delta, terminated, _, info = env.reset()
            agent = AgentMDPPlanning(parameter_values, world=env)
            t = 0
            total = delta
            running = True
            stepping = False
        elif event == 'pause_button':
            print("PAUSE")
            running = False
            stepping = False
        elif event == 'step_button':
            print("STEP")
            running = False
            stepping = True
        elif event == 'continue_button':
            print("CONTINUE")
            running = True
            stepping = False
        else:
            parsed_events = True

    if env and (running or stepping) and not terminated:
        env._fps = values['speed_slider']
        action, aleph4action = agent.localPolicy(state, aleph).sample()[0]
        print("t:",t, ", last delta:",delta, ", total:", total, ", s:",state, ", aleph4s:", aleph, ", a:", action, ", aleph4a:", aleph4action)
        nextState, delta, terminated, _, info = env.step(action)
        total += delta
        aleph = agent.propagateAspiration(state, action, aleph4action, delta, nextState)
        state = nextState
        if terminated:
            print("t:",t, ", last delta:",delta, ", final total:", total, ", final s:",state, ", aleph4s:", aleph)
            print("Terminated.")
            running = stepping = False
        t += 1
        if stepping: stepping = False
    else:
        time.sleep(0.1)

window.close()

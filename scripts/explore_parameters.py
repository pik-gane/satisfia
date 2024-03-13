import sys
sys.path.insert(0,'./src/')

import time
import PySimpleGUI as sg

from environments.very_simple_gridworlds import make_simple_gridworld
from satisfia.agents.makeMDPAgentSatisfia import AgentMDPPlanning

gridworlds = ["AISG2", "GW1", "GW2", "GW3", "GW4", "GW5", "GW6", "GW22", "test_box"]
parameter_data = [
    ("lossTemperature", 0, 100, 1, 1),
    ("lossCoeff4Variance", -100, 100, 0, 1),

    ('lossCoeff4Fourth', -100, 100, 0, 1), 
    ('lossCoeff4Cup', -100, 100, 0, 1), 

    ('lossCoeff4FeasibilityPower', -100, 100, 0, 1), 
    ('lossCoeff4DP', -100, 100, 0, 1), 

    ('lossCoeff4LRA1', -100, 100, 0, 1), 
    ('lossCoeff4LRA', -100, 100, 0, 1), 

    ('lossCoeff4Time1', -100, 100, 0, 1), 
    ('lossCoeff4Time', -100, 100, 0, 1), 

    ('lossCoeff4Entropy1', -100, 100, 0, 1), 
    ('lossCoeff4Entropy', -100, 100, 0, 1), 

    ('lossCoeff4KLdiv1', -100, 100, 0, 1), 
    ('lossCoeff4KLdiv', -100, 100, 0, 1),

    ('lossCoeff4DeltaVariation', -100, 100, 0, 1), 
    ('lossCoeff4TrajectoryEntropy', -100, 100, 0, 1), 

#    ('lossCoeff4Random', -100, 100, 0, 1), 

    ('minLambda', 0, 1, 0, 0.01), 
    ('maxLambda', 0, 1, 1, 0.01), 
] # name, min, max, initial

class policy():
    def __init__(self):
        pass
    def __call__(self, state):
        return self
    def score(self, action):
        return 1
uninformedPolicy = policy()

# Create a drop down for selecting the gridworld
gridworld_dropdown = sg.DropDown(gridworlds, default_value=gridworlds[0], key='gridworld_dropdown')

# Create "verbose" and "debug" toggles:
verbose_checkbox = sg.Checkbox("Verbose", default=False, key='verbose_checkbox')
debug_checkbox = sg.Checkbox("Debug", default=False, key='debug_checkbox')

# Create a "reset" button for resetting all parameter values to their defaults:
reset_button = sg.Button("Reset", key='reset_button')

# Create sliders for setting the parametersers
parameter_sliders = {}
for pd in parameter_data:
    parameter_sliders[pd[0]] = sg.Slider(range=(pd[1], pd[2]), default_value=pd[3], resolution=pd[4], orientation='h', key=pd[0])


# Create buttons for starting, pausing, stepping, and continuing the simulation
restart_button = sg.Button("(Re)start", key='restart_button')
pause_button = sg.Button("Pause", key='pause_button')
step_button = sg.Button("Step", key='step_button')
continue_button = sg.Button("Continue", key='continue_button')
speed_slider = sg.Slider(range=(1, 20), default_value=10, orientation='h', key='speed_slider')

# Create the layout
s = max([len(pd[0]) for pd in parameter_data])
layout = [
    [sg.Text("Gridworld"), gridworld_dropdown, verbose_checkbox, debug_checkbox, reset_button],
    [sg.Column([
        [
            sg.Text(parameter_data[2*r][0], size=(s,None), justification="right"), parameter_sliders[parameter_data[2*r][0]],
            sg.Text(parameter_data[2*r+1][0], size=(s,None), justification="right"), parameter_sliders[parameter_data[2*r+1][0]],
        ]
        for r in range(len(parameter_data) // 2)
        ], element_justification='r')],
    [restart_button, pause_button, step_button, continue_button], 
    [sg.Text("Speed"), speed_slider]
]

# Create the window

window = sg.Window("SatisfIA Control Panel", layout, location=(0,0))

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
        elif event == 'reset_button':
            for pd in parameter_data:
                window[pd[0]].update(pd[3])
        elif event == 'restart_button':
            gridworld = values['gridworld_dropdown']
            parameter_values = { pd[0]: values[pd[0]] for pd in parameter_data }
            parameter_values.update({
                'verbose': values['verbose_checkbox'],
                'debug': values['debug_checkbox'],
                'allowNegativeCoeffs': True,
                'uninformedPolicy': uninformedPolicy
            })
            print("\n\nRESTART gridworld", gridworld, parameter_values)
            env, aleph = make_simple_gridworld(gw=gridworld, render_mode="human", fps=values['speed_slider'])
            state, delta, terminated, _, info = env.reset()
            agent = AgentMDPPlanning(parameter_values, world=env)
            t = 0
            total = delta
            running = True
            stepping = False
        elif event == 'pause_button':
            print("\n\nPAUSE")
            running = False
            stepping = False
        elif event == 'step_button':
            print("\n\nSTEP")
            running = False
            stepping = True
        elif event == 'continue_button':
            print("\n\nCONTINUE")
            running = True
            stepping = False
        else:
            parsed_events = True

    if env and (running or stepping) and not terminated:
        env._fps = values['speed_slider']
        action, aleph4action = agent.localPolicy(state, aleph).sample()[0]
        if parameter_values['verbose'] or parameter_values['debug']:
            print("t:",t, ", last delta:",delta, ", total:", total, ", s:",state, ", aleph4s:", aleph, ", a:", action, ", aleph4a:", aleph4action)
        nextState, delta, terminated, _, info = env.step(action)
        total += delta
        aleph = agent.propagateAspiration(state, action, aleph4action, delta, nextState)
        state = nextState
        if terminated:
            if parameter_values['verbose'] or parameter_values['debug']:
                print("t:",t, ", last delta:",delta, ", final total:", total, ", final s:",state, ", aleph4s:", aleph)
                print("Terminated.")
            running = stepping = False
        t += 1
        if stepping: stepping = False
    else:
        time.sleep(0.1)

window.close()

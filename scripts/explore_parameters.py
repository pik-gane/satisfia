#!/usr/bin/env python3

import sys
sys.path.insert(0,'./src/')

import time
import PySimpleGUI as sg

from environments.very_simple_gridworlds import make_simple_gridworld, all_worlds
from satisfia.agents.makeMDPAgentSatisfia import AgentMDPPlanning

gridworlds = sorted(all_worlds())
default_gridworld = "test_return"

parameter_data = [
    ("aleph0_low", -10, 10, 0, 0.1),
    ("aleph0_high", -10, 10, 0, 0.1),

    ("lossTemperature", 0, 100, 0, 1),
    ("lossCoeff4Variance", -100, 100, 0, 1),

    ('lossCoeff4Fourth', -100, 100, 0, 1), 
    ('lossCoeff4Cup', -100, 100, 0, 1), 

    ('lossCoeff4WassersteinTerminalState', -100, 100, 100, 1), 
    ('lossCoeff4AgencyChange', -100, 100, 0, 1), 

    ('lossCoeff4StateDistance', -100, 100, 0, 1), 
    ('lossCoeff4Causation', -100, 100, 0, 1), 

 #   ('lossCoeff4CausationPotential', -100, 100, 0, 1), 
#    ('lossCoeff4Random', -100, 100, 0, 1), 

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

    ('minLambda', 0, 1, 0, 0.01), 
    ('maxLambda', 0, 1, 1, 0.01), 
] # name, min, max, initial, step-size

class policy():
    def __init__(self):
        pass
    def __call__(self, state):
        return self
    def score(self, action):
        return 1
uninformedPolicy = policy()

# Create a drop down for selecting the gridworld
gridworld_dropdown = sg.DropDown(gridworlds, default_value=default_gridworld, key='gridworld_dropdown')

override_aleph_checkbox = sg.Checkbox("Override aleph0", default=False, key='override_aleph_checkbox', enable_events = True)

# Create "verbose" and "debug" toggles:
verbose_checkbox = sg.Checkbox("Verbose", default=False, key='verbose_checkbox')
debug_checkbox = sg.Checkbox("Debug", default=False, key='debug_checkbox')

# Create a "reset" button for resetting all parameter values to their defaults:
reset_params_button = sg.Button("Reset parameters", key='reset_params_button')

# Create sliders for setting the parametersers
parameter_sliders = {}
for pd in parameter_data:
    parameter_sliders[pd[0]] = sg.Slider(range=(pd[1], pd[2]), default_value=pd[3], resolution=pd[4], orientation='h', key=pd[0], 
                                         disabled = pd[0] in ['aleph0_low', 'aleph0_high'])

# Create buttons for starting, pausing, stepping, and continuing the simulation
reset_env_button = sg.Button("Reset", key='reset_env_button')
restart_button = sg.Button("Restart", key='restart_button')
pause_button = sg.Button("Pause", key='pause_button')
step_button = sg.Button("Step", key='step_button')
continue_button = sg.Button("Start/Continue", key='continue_button')

autorestart_checkbox = sg.Checkbox("Auto restart", default=True, key='autorestart_checkbox')

speed_slider = sg.Slider(range=(1, 20), default_value=10, orientation='h', key='speed_slider')

# Create the layout
state = max([len(pd[0]) for pd in parameter_data])
layout = [
    [sg.Text("Gridworld"), gridworld_dropdown, override_aleph_checkbox, verbose_checkbox, debug_checkbox, reset_params_button],
    [sg.Column([
        [
            sg.Text(parameter_data[2*r][0], size=(state,None), justification="right"), parameter_sliders[parameter_data[2*r][0]],
            sg.Text(parameter_data[2*r+1][0], size=(state,None), justification="right"), parameter_sliders[parameter_data[2*r+1][0]],
        ]
        for r in range(len(parameter_data) // 2)
        ], element_justification='r')],
    [sg.Text("Simulation:"),
     reset_env_button, restart_button, pause_button, step_button, continue_button], 
    [autorestart_checkbox, sg.Text("Speed"), speed_slider]
]

# Create the window

window = sg.Window("SatisfIA Control Panel", layout, location=(0,0))

gridworld = None
parameter_values = { pd[0]: pd[3] for pd in parameter_data }
env = None
agent = None
running = False
stepping = False
terminated = False

def step():
    global gridworld, parameter_values, env, agent, running, stepping, terminated, t, state, total, aleph, aleph0, delta, initialMu0, initialMu20, visited_state_alephs, visited_action_alephs
    print()
    env._fps = values['speed_slider']
    action, aleph4action = agent.localPolicy(state, aleph).sample()[0]
    visited_state_alephs.add((state, aleph))
    visited_action_alephs.add((state, action, aleph4action))
    if values['lossCoeff4WassersteinTerminalState'] != 0:
        print("  in state", state)
        for a in agent.world.possible_actions(state):
            al4a = agent.aspiration4action(state, a, aleph)
            print("    taking action", a, "gives:")
            print("      default ETerminalState_state (s0):", initialMu0)
            print("      default ETerminalState2_state(s0):", initialMu20)
            print("      actual  ETerminalState_state (s) :", list(agent.ETerminalState_action(state, a, al4a, "actual")))
            print("      actual  ETerminalState2_state(s) :", list(agent.ETerminalState2_action(state, a, al4a, "actual")))
            print("      --> Wasserstein distance", agent.wassersteinTerminalState_action(state, a, al4a))
            print("      expected Total (state aleph):", agent.Q(state, a, al4a), f"({aleph})")
        print("    so we take action", action)
    if parameter_values['verbose'] or parameter_values['debug']:
        print("t:", t, ", last delta:" ,delta, ", total:", total, ", s:", state, ", aleph4s:", aleph, ", a:", action, ", aleph4a:", aleph4action)
    nextState, delta, terminated, _, info = env.step(action)
    total += delta
    aleph = agent.propagateAspiration(state, action, aleph4action, delta, nextState)
    state = nextState
    if terminated:
        print("t:",t, ", last delta:",delta, ", final total:", total, ", final s:", state, ", aleph4s:", aleph)
        print("Terminated.")
        running = stepping = False
        if values['autorestart_checkbox']:
            time.sleep(0.2)
            reset_env(True)
            env.render()
        elif values['debug_checkbox'] or values['verbose_checkbox']:
            if values['debug_checkbox']:
                visited_state_alephs = agent.seen_state_alephs.copy()
                visited_action_alephs = agent.seen_action_alephs.copy()
            Vs = {}
            for state, aleph in visited_state_alephs:
                t, loc, imm_states, mc_locs, mv_locs, mv_states = env._extract_state_attributes(state)
                if loc not in Vs:
                    Vs[loc] = []
                Vs[loc].append(f"{aleph[0]},{aleph[1]}:{agent.V(state, aleph)}")  # expected Total
            Qs = {}
            for state, action, aleph in visited_action_alephs:
                t, loc, imm_states, mc_locs, mv_locs, mv_states = env._extract_state_attributes(state)
                key = (*loc, action)
                if key not in Qs:
                    Qs[key] = []
#                    Qs[key].append(agent.Q(state, action, aleph))
                Qs[key].append(f'{aleph[0]},{aleph[1]}:{agent.Q(state, action, aleph)}')  # variance of Total
            
            env.render(additional_data={
                'cell': Vs,
                'action' : Qs
            })
    else:
        print("t:",t, ", delta:",delta, ", total:", total, ", s:", state, ", aleph4s:", aleph)
        t += 1
        if stepping: stepping = False

def reset_env(start=False):
    # TODO: only regenerate env if different from before!
    global gridworld, parameter_values, env, agent, running, stepping, terminated, t, state, total, aleph, aleph0, delta, initialMu0, initialMu20, visited_state_alephs, visited_action_alephs
    old_gridworld = gridworld
    gridworld = values['gridworld_dropdown']
    if gridworld != old_gridworld:
        env, aleph0 = make_simple_gridworld(gw=gridworld, render_mode="human", fps=values['speed_slider'])
#        env = env.get_prolonged_version(5)
    if values['override_aleph_checkbox']:
        aleph = (values['aleph0_low'], values['aleph0_high'])
    else:
        aleph = aleph0
        parameter_sliders['aleph0_low'].update(aleph[0])
        parameter_sliders['aleph0_high'].update(aleph[1])
    parameter_values = { pd[0]: values[pd[0]] for pd in parameter_data }
    if parameter_values['lossTemperature'] == 0:
        parameter_values['lossTemperature'] = 1e-6
    parameter_values.update({
        'verbose': values['verbose_checkbox'],
        'debug': values['debug_checkbox'],
        'allowNegativeCoeffs': True,
        'uninformedPolicy': uninformedPolicy,
        'referenceState': env.initial_state()
    })
    print("\n\nRESTART gridworld", gridworld, parameter_values)
    state, info = env.reset()
    print("Initial state:", env.state_embedding(state), ", initial aleph:", aleph)
    agent = AgentMDPPlanning(parameter_values, world=env)
    # agent.localPolicy(state, aleph)  # call it once to precompute tables and save time for later
    initialMu0 = list(agent.ETerminalState_state(state, aleph, "default"))
    initialMu20 = list(agent.ETerminalState2_state(state, aleph, "default"))
    t = 0
    delta = total = 0
    terminated = False
    visited_state_alephs = set()
    visited_action_alephs = set()
    running = start
    stepping = False

wait = time.monotonic()
while True:
    event, values = window.read(timeout=0)
    if event != '__TIMEOUT__': 
        print(event)
    if event == sg.WINDOW_CLOSED:
        break
    elif event == 'reset_params_button':
        for pd in parameter_data:
            window[pd[0]].update(pd[3])
    elif event == 'reset_env_button':
        reset_env(False)
    elif event == 'restart_button':
        reset_env(True)
    elif event == 'pause_button':
        print("\n\nPAUSE")
        running = False
    elif event == 'step_button':
        print("\n\nSTEP")
        step()
    elif event == 'continue_button':
        print("\n\nCONTINUE")
        running = True
    elif event == 'override_aleph_checkbox':
        parameter_sliders['aleph0_low'].update(disabled=not values['override_aleph_checkbox'])
        parameter_sliders['aleph0_high'].update(disabled=not values['override_aleph_checkbox'])
    elif running and  (time.monotonic() - wait) > 1/values['speed_slider'] and not terminated:
        step()
        wait = time.monotonic()
    elif event == '__TIMEOUT__':
        time.sleep(.1)                         

window.close()

# satisfia

Intelligent agents that satisfy aspirations. This is the main repository of code produced in the [SatisfIA project](https://pik-gane.github.io/satisfia/).

![image](https://github.com/pik-gane/satisfia/assets/22815964/d2c2e297-eb51-4cef-8989-875e8bf20719)


## Requirements

tested with:
- python 3.10.0
- gymnasium 0.29.0
- pygame 2.5.0
- mujoco-py 2.1.2.14

## Getting started

### Running around a very simple gridworld and a larger random gridworld:
```
python scripts/test_simple_gridworld.py
```

### A GUI for exploring parameters:
```
python scripts/explore_parameters.py
```

### Adding more gridworlds:
Open `src/environments/very_simple_gridworlds.py` and add another `elif` section.
 

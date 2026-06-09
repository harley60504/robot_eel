"""Default MuJoCo start-position settings.

The 3 m tank is still represented in MuJoCo world coordinates as x=-1.5..1.5.
Using x=-1.25 places the eel 0.25 m inside the left boundary, which corresponds
to 0.25 m from the 0 m start when reading the tank as 0..3 m.
"""

DEFAULT_START_X = -1.25
DEFAULT_START_Y = 0.0

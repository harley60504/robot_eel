# XML + Python config split v2

Core files:
- water_settings.xml: central water density, viscosity, gravity, and eel fluidcoef.
- environment_3x1_5.xml: field / tank only.
- eel_body.xml: shared eel body only.
- eel.xml: main wrapper that includes the water, field, and eel body.
- sim_config.py: shared Python defaults.

Rectangle mode:
- Uses the same eel.xml body.
- Uses RECTANGLE_CONTROL_SIGN = -1.0 to preserve the old eel_rectangle steering direction.
- If steering direction is opposite, change RECTANGLE_CONTROL_SIGN to 1.0 in sim_config.py.

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent

EEL_MODEL_XML = str(ROOT / "eel.xml")
RECTANGLE_MODEL_XML = str(ROOT / "eel.xml")
TETHERED_MODEL_XML = str(ROOT / "eel_tethered.xml")

TANK_X_MIN = 0.0
TANK_X_MAX = 3.0
TANK_CENTER_X = 1.5
TANK_Y_HALF = 0.75

DEFAULT_START_X = 0.60
DEFAULT_START_Y = 0.0

# Rectangle-course centerline. The long-side walls are y = +/-0.75 m, and the
# eel body/tail can swing far outside the base_link path while turning. Keep the
# y centerline narrow and use more of the x length so the course stays visibly
# rectangular without scraping the long-side walls.
RECTANGLE_WAYPOINTS = "2.75,-0.12;2.75,0.12;0.25,0.12;0.25,-0.12"
RECTANGLE_PATH_HALF_X = 1.25
RECTANGLE_PATH_HALF_Y = 0.12
RECTANGLE_PATH_CENTER_X = 1.50
RECTANGLE_PATH_CENTER_Y = 0.0

# Old eel_rectangle.xml used the opposite servo axis.
# Rectangle mode now uses the same eel.xml body, so this preserves old steering behavior.
RECTANGLE_CONTROL_SIGN = -1.0

RESET_X_MIN = -0.225
RESET_X_MAX = 3.225
RESET_Y = 0.90

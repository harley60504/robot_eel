# Legacy real-eel tracking files and data

This directory is kept as a root-level archive for the old real-video recognition workflow.

The active videos are expected under the local Windows path:

```text
C:\Users\ytyla\Documents\GitHub\robot_eel\Release\python_backend\recordings
```

Main recording clips:

```text
clean_v_20260607_233739.mp4  # straight swim
clean_v_20260608_141203.mp4  # left turn
clean_v_20260608_141254.mp4  # spin left
```

## Legacy scripts

The old analysis scripts were separated by recognition target:

```text
analyze_eel_video_shape.py        # whole body / silicone shape detector
analyze_robot_composite_video.py  # white body + black servo + red/orange mark detector
analyze_servo_blocks_video.py     # black servo-block chain detector
make_tracked_center_cleaned_physical.py  # old red-marker tracker for video_analysis JSON
```

## Why this is legacy

The old `make_tracked_center_cleaned_physical.py` used a full-frame red-mask detector and selected the largest red contour. On some videos it could lock on to timestamp/text/noise near the top-right of the frame, producing wrong points such as `x ~= 1000..1077, y ~= 12..25` and a false radius near `37 px`.

That bad result is not the real eel trajectory. The plotted fit curve for the real left turn should be around `R ~= 0.43..0.44 m` when using the original scale `875 px / 1.5 m`.

## Current replacement

The active tracker is:

```text
mujoco_simulation/make_tracked_center_cleaned_physical.py
```

It now uses a merged recognition strategy:

```text
composite detector first  -> white body + dark servo/electronics + red/orange marks
legacy red detector next  -> fallback for frames where composite detection fails
continuity scoring        -> rejects sudden jumps and timestamp/noise detections
ROI cropping              -> excludes timestamp/rim/borders
```

The output format is intentionally kept compatible:

```text
outputs/video_analysis/<video_stem>/tracked_center_summary_cleaned_physical.json
```

so `plot_fitted_gait_curves.py` can still read `cleaned_points` and compute/plot the fitted radius.

## Regenerate current data

From `mujoco_simulation`:

```powershell
python make_tracked_center_cleaned_physical.py
python plot_fitted_gait_curves.py
```

To test only one file:

```powershell
python make_tracked_center_cleaned_physical.py --videos clean_v_20260607_233739.mp4
```

To compare detector modes:

```powershell
python make_tracked_center_cleaned_physical.py --detector merged
python make_tracked_center_cleaned_physical.py --detector composite
python make_tracked_center_cleaned_physical.py --detector red
```

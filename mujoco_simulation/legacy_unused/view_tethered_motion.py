from __future__ import annotations

import time

import mujoco
import mujoco.viewer
import numpy as np

from hopf_cpg import HopfCPG, HopfCPGParams


XML_PATH = "eel_tethered.xml"
AJOINT = 0.45
FREQ = 1.0
WAVELENGTH = 1.5


def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    model.opt.gravity[:] = (0, 0, 0)
    cpg = HopfCPG(
        num_joints=6,
        params=HopfCPGParams(frequency=FREQ, wavelength=WAVELENGTH, ajoint=AJOINT),
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = np.array([-1.2, 0.0, -0.02])
            viewer.cam.distance = 1.2
            viewer.cam.elevation = -70
            viewer.cam.azimuth = 0

        while viewer.is_running():
            data.ctrl[0:3] = 0.0
            data.ctrl[3:9] = np.clip(cpg.step(data.time, model.opt.timestep), -1.2, 1.2)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()

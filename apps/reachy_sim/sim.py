"""Pollen's MuJoCo sim of Reachy Mini as a test layer.

`python sim.py PORT [SCENE]` runs the sim daemon headless on 127.0.0.1:PORT,
never advertised on the LAN. SCENE is an SDK scene (empty, minimal) or one in
scenes/ here (desk).

The headless daemon renders no camera (Pollen only starts the render thread with
the viewer), so `eye_camera` renders the eye in-process, offscreen, from joints
read back from the daemon.
"""

import os
import shutil
import sys
from functools import cache
from importlib.resources import files
from pathlib import Path

import mujoco
import numpy as np
import reachy_mini

SDK_SCENES = Path(str(files(reachy_mini).joinpath("descriptions/reachy_mini/mjcf/scenes")))
# A quarter of the real 1280x720, same aspect: cheap on the CPU side.
EYE_SIZE = (180, 320)
SETTLE_STEPS = 1500  # 3 s of sim at 500 Hz: the stewart platform's passive joints close


def install_scenes() -> None:
    """Copy our scenes next to the SDK's, the only place its daemon loads scenes from."""
    # ponytail: writes into our own .venv; an upstream --scene-path would remove this.
    for scene in (Path(__file__).parent / "scenes").glob("*.xml"):
        shutil.copy(scene, SDK_SCENES / scene.name)


@cache
def _model(scene: str) -> mujoco.MjModel:
    install_scenes()
    return mujoco.MjModel.from_xml_path(str(SDK_SCENES / f"{scene}.xml"))


@cache
def _renderer(scene: str) -> mujoco.Renderer:
    return mujoco.Renderer(_model(scene), *EYE_SIZE)


def geom_id(scene: str, name: str) -> int:
    """Return the id a geom has in segmentation frames of this scene."""
    return _model(scene).geom(name).id


def eye_camera(
    head_joints: list[float], antennas: list[float], scene: str = "minimal", segment: bool = False
) -> np.ndarray:
    """Render the eye camera with the robot at these joints, as the daemon reports them.

    RGB uint8 (h, w, 3), or with `segment` the geom id per pixel in [..., 0] (-1: none).
    """
    model = _model(scene)
    data = mujoco.MjData(model)
    # The daemon's actuator order: yaw_body, stewart_1..6, then antennas sign-flipped.
    data.ctrl[:7] = head_joints
    data.ctrl[-2:] = -np.asarray(antennas)
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)
    renderer = _renderer(scene)
    if segment:
        renderer.enable_segmentation_rendering()
    else:
        renderer.disable_segmentation_rendering()
    renderer.update_scene(data, "eye_camera")
    return renderer.render()


def main(port: str, scene: str = "minimal") -> None:
    """Run Pollen's daemon in sim, headless, on localhost only."""
    import reachy_mini.utils.discovery
    from reachy_mini.daemon.app.main import main as daemon

    install_scenes()
    # Recorded moves come from the local cache only: the sim never downloads.
    os.environ["HF_HUB_OFFLINE"] = "1"
    # The daemon advertises itself over mDNS unconditionally: a sim must never
    # be discoverable as a robot.
    reachy_mini.utils.discovery.MdnsServiceRegistration.register = lambda self: None
    sys.argv = [
        "reachy-mini-daemon",
        "--sim",
        "--headless",
        "--no-media",
        "--scene",
        scene,
        "--robot-name",
        "reachy_sim",
        "--fastapi-host",
        "127.0.0.1",
        "--fastapi-port",
        port,
        "--no-preload-datasets",
        "--dataset-update-interval",
        "0",
    ]
    daemon()


if __name__ == "__main__":
    main(*sys.argv[1:])

#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run a trained wheelbipe policy in MuJoCo and control it through MJViser."""

from __future__ import annotations

import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import viser  # noqa: E402
from mjviser import Viewer  # noqa: E402

from supermimic_wheelbipe.constants import ACTION_CLIP, default_term_height  # noqa: E402
from supermimic_wheelbipe.command import CommandLib  # noqa: E402
from supermimic_wheelbipe.mdp import MDP  # noqa: E402
from supermimic_wheelbipe.mujoco_env import DEFAULT_XML, MujocoWheelbipe, load_policy_ckpt  # noqa: E402

VX_LIMIT = 1.5
WZ_LIMIT = 10.21
TERM_COS = float(np.cos(np.deg2rad(40.0)))


def main() -> int:
    parser = argparse.ArgumentParser(description="MJViser policy replay and browser control")
    parser.add_argument("--ckpt", default="logs/mjlab_4096_30000/ckpt_030000.pt")
    parser.add_argument("--xml", default=DEFAULT_XML)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default="cpu")
    args_cli = parser.parse_args()

    checkpoint = args_cli.ckpt
    if not os.path.isabs(checkpoint):
        checkpoint = os.path.join(REPO, checkpoint)
    policy, spec, train_args = load_policy_ckpt(checkpoint, args_cli.device)
    goal_frames = int(train_args["goal_frames"])
    history_length = int(train_args["hist"])
    env = MujocoWheelbipe(spec, xml=args_cli.xml, device=args_cli.device)
    contact_height, stand_height = env.settle_stand_height()
    height_limits = tuple(train_args.get("height_range") or (0.16, 0.28))
    term_height = train_args.get("term_height")
    if term_height is None:
        term_height = default_term_height(stand_height)

    command = CommandLib.fixed(
        [[0.0, 0.0, 0.0, sum(height_limits) / 2]],
        device=args_cli.device,
        frames=goal_frames,
    )
    mdp = MDP(spec, command, kin=None, default_q=torch.zeros(spec.nj, device=args_cli.device), qd_limit=None)
    policy_state = {
        "history": None,
        "previous_action": torch.zeros(1, spec.na, device=args_cli.device),
        "control_steps": 0,
        "reset_requested": False,
        "forward_command": 0.0,
        "yaw_command": 0.0,
    }

    server = viser.ViserServer(host=args_cli.host, port=args_cli.port, label="Wheelbipe Policy Control")
    forward_slider = server.gui.add_slider(
        "Forward speed (m/s)", min=-VX_LIMIT, max=VX_LIMIT, step=0.05, initial_value=0.0
    )
    yaw_slider = server.gui.add_slider(
        "Yaw speed (rad/s)", min=-WZ_LIMIT, max=WZ_LIMIT, step=0.1, initial_value=0.0
    )
    height_slider = server.gui.add_slider(
        "Target height (m)", min=height_limits[0], max=height_limits[1], step=0.005,
        initial_value=sum(height_limits) / 2,
    )
    telemetry = server.gui.add_markdown("Loading policy state…")
    reset_button = server.gui.add_button("Reset robot")

    @reset_button.on_click
    def _request_reset(_event) -> None:
        policy_state["reset_requested"] = True

    control_steps_per_action = env.decim

    def reset_simulation(_model, _data) -> None:
        env.reset(z=stand_height)
        policy_state["history"] = None
        policy_state["previous_action"] = torch.zeros(1, spec.na, device=args_cli.device)
        policy_state["control_steps"] = 0
        policy_state["forward_command"] = 0.0
        policy_state["yaw_command"] = 0.0
        command.cmd[:] = command.cmd.new_tensor((0.0, 0.0, 0.0, height_slider.value))

    def policy_step(model, data) -> None:
        if policy_state["reset_requested"]:
            policy_state["reset_requested"] = False
            reset_simulation(model, data)

        if policy_state["control_steps"] % control_steps_per_action == 0:
            target_forward = float(forward_slider.value)
            target_yaw = float(yaw_slider.value)
            policy_state["forward_command"] += float(
                np.clip(target_forward - policy_state["forward_command"], -0.04, 0.04)
            )
            policy_state["yaw_command"] += float(
                np.clip(target_yaw - policy_state["yaw_command"], -0.2, 0.2)
            )
            command.cmd[:] = command.cmd.new_tensor(
                (policy_state["forward_command"], 0.0, policy_state["yaw_command"], float(height_slider.value))
            )

            state = env.state()
            observation = mdp.obs(state, policy_state["previous_action"])
            history = policy_state["history"]
            policy_state["history"] = (
                observation.unsqueeze(1).repeat(1, history_length, 1)
                if history is None
                else torch.cat((history[:, 1:], observation.unsqueeze(1)), dim=1)
            )
            with torch.no_grad():
                action = policy(
                    policy_state["history"], command.cmd.reshape(1, -1)
                ).clamp(-ACTION_CLIP, ACTION_CLIP)
            env.apply_action(action)
            policy_state["previous_action"] = action

        mujoco.mj_step(model, data)
        policy_state["control_steps"] += 1

        z = env.base_z
        if z < term_height or env.tilt_cos < TERM_COS or not np.isfinite(z):
            reset_simulation(model, data)

        if policy_state["control_steps"] % (control_steps_per_action * 10) == 0:
            telemetry.content = (
                f"**Policy iteration** {train_args['iteration']}  \n"
                f"**Forward** {env.vel_forward:+.2f} / {policy_state['forward_command']:+.2f} m/s  \n"
                f"**Yaw rate** {data.qvel[5]:+.2f} / {policy_state['yaw_command']:+.2f} rad/s  \n"
                f"**Height** {z:.3f} / {height_slider.value:.3f} m"
            )

    def render_scene(scene) -> None:
        scene.update_from_mjdata(env.d)

    viewer = Viewer(
        env.m,
        env.d,
        step_fn=policy_step,
        render_fn=render_scene,
        reset_fn=reset_simulation,
        server=server,
    )
    print(
        f"[viser] checkpoint={checkpoint} iteration={train_args['iteration']} "
        f"contact_z={contact_height:.4f} stand_z={stand_height:.4f} "
        f"url=http://127.0.0.1:{args_cli.port}",
        flush=True,
    )
    viewer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

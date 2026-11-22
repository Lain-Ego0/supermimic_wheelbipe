#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""在 MuJoCo 中以固定指令评测 Isaac 训练出的策略。"""

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from supermimic_wheelbipe.command import CommandLib, eval_pairs  # noqa: E402
from supermimic_wheelbipe.constants import ACTION_CLIP, default_term_height  # noqa: E402
from supermimic_wheelbipe.kinematics import Kinematics  # noqa: E402
from supermimic_wheelbipe.mdp import MDP, MAX_STEP_REWARD, RewardCfg  # noqa: E402
from supermimic_wheelbipe.model import load_model, wheel_bodies  # noqa: E402
from supermimic_wheelbipe.mujoco_env import DEFAULT_XML, MujocoWheelbipe, load_policy_ckpt  # noqa: E402


@torch.inference_mode()
def evaluate(ckpt, xml, model_json, height, steps, term_height):
    policy, spec, train_args = load_policy_ckpt(str(ckpt))
    env = MujocoWheelbipe(spec, xml=str(xml))
    _, stand_height = env.settle_stand_height()
    if term_height is None:
        term_height = train_args.get("term_height")
    if term_height is None:
        term_height = default_term_height(stand_height)
    if height is None:
        limits = train_args.get("height_range") or (0.16, 0.28)
        height = sum(limits) / 2

    pairs = eval_pairs()
    commands = CommandLib.fixed([[vx, 0.0, wz, height] for vx, wz in pairs], "cpu", frames=steps + 1)
    model = load_model(model_json)
    kin = Kinematics(model, spec.joint_names, "cpu", wheel_bodies=wheel_bodies(model, spec))
    mdp = MDP(spec, commands, kin=kin, cfg=RewardCfg(term_height=term_height))
    goal_frames = train_args["goal_frames"]
    history_length = train_args["hist"]
    cases = []
    for index, (vx, wz) in enumerate(pairs):
        env.reset(z=stand_height)
        history = None
        previous_action = torch.zeros(1, spec.na)
        # 每条评测指令恒定，前视窗口也始终相同。
        frame = commands.clip_start[index:index + 1]
        goal = commands.goal(frame, goal_frames)
        reward_sum = 0.0
        error_sum = torch.zeros(3)
        terminated = False
        state = env.state()
        for step in range(1, steps + 1):
            observation = mdp.obs(state, previous_action).unsqueeze(1)
            if history is None:
                history = observation.repeat(1, history_length, 1)
            else:
                history = torch.cat((history[:, 1:], observation), dim=1)
            action = policy(history, goal).clamp(-ACTION_CLIP, ACTION_CLIP)
            state = env.step(action)
            reward, terms = mdp.reward(state, frame, action, previous_action, terms=True)
            errors = torch.cat([terms[key].abs() for key in ("e_lin", "e_ang", "e_height")])
            if not torch.isfinite(reward).all() or not torch.isfinite(errors).all():
                raise RuntimeError(f"第 {index + 1} 条指令在第 {step} 步产生非有限状态")
            reward_sum += reward.item()
            error_sum += errors
            previous_action = action
            terminated = bool(terms["terminated"].item())
            if terminated:
                break
        linear, angular, height_error = (error_sum / step).tolist()
        cases.append({
            "vx": vx, "wz": wz, "steps": step, "failed": terminated,
            "norm_return": reward_sum / (steps * MAX_STEP_REWARD),
            "mean_abs_e_lin": linear, "mean_abs_e_ang": angular, "mean_abs_e_h": height_error,
        })
        print(f"vx={vx:+.2f} wz={wz:+.2f}  {step}/{steps} 步  "
              f"{'终止' if terminated else '通过'}  前向误差 {linear:.3f}  偏航误差 {angular:.3f}", flush=True)

    averages = {
        key: sum(case[key] for case in cases) / len(cases)
        for key in ("norm_return", "mean_abs_e_lin", "mean_abs_e_ang", "mean_abs_e_h")
    }
    return {
        "checkpoint": str(ckpt.resolve()), "iteration": train_args["iteration"],
        "height_cmd": height, "stand_height": stand_height, "term_height": term_height,
        "control_dt": env.control_dt, "steps_per_case": steps,
        "n_failed": sum(case["failed"] for case in cases), **averages, "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=REPO / "logs/wb_final/last.pt")
    parser.add_argument("--xml", type=Path, default=Path(DEFAULT_XML))
    parser.add_argument("--model-json", type=Path, default=REPO / "data/wheelbipe_v14_2_model.json")
    parser.add_argument("--height-cmd", type=float)
    parser.add_argument("--term-height", type=float)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--out", type=Path, help="可选的评测结果 JSON 路径")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps 必须大于 0")
    torch.set_num_threads(1)
    result = evaluate(args.ckpt, args.xml, args.model_json, args.height_cmd, args.steps, args.term_height)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))
    return int(result["n_failed"] > 0)


if __name__ == "__main__":
    sys.exit(main())

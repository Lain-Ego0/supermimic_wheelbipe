# SPDX-License-Identifier: MIT
"""wheelbipe V14_2 机器人：资产定位、ArticulationCfg、关节分组（RobotSpec）、动作尺度、弹簧力模型。"""

from __future__ import annotations

import os
import re


from .constants import (  # noqa: F401  (re-export，其他模块从这里拿常量)
    ACTION_CLIP,
    ACTION_SCALE_LEG,
    ACTION_SCALE_WHEEL,
    CONTROL_DT,
    DECIMATION,
    SIM_DT,
    SPRING_BIAS_RANGE,
    SPRING_FORCE_DOWN,
    SPRING_FORCE_UP,
    SPRING_LENGTH,
    SPRING_OFFSET,
    WHEEL_RADIUS,
    WHEEL_EFFORT,
    WHEEL_VEL_KD,
    WHEEL_VEL_LIMIT,
    spring_force,
)

# 资产
# 可以用环境变量覆盖；否则按常见目录顺序查找
_ASSET_SUBDIR = os.path.join("source", "agent_world", "agent_world", "assets", "usd_files")


def _candidates() -> list[str]:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = []
    env = os.environ.get("WHEELBIPE_ASSET_ROOT")
    if env:
        out.append(env)
    out += [
        os.path.join(here, "..", "wheeled-legged_RL", _ASSET_SUBDIR),
        os.path.join(here, "data", "wheeled-legged_RL", _ASSET_SUBDIR),
        os.path.join(here, "assets"),
    ]
    try:  # 如果 wheeled-legged_RL 已经 pip install -e，直接用它的 AssetPath
        import agent_world  # noqa: F401

        out.insert(0, os.path.join(str(agent_world.AssetPath), "usd_files"))
    except Exception:
        pass
    return [os.path.normpath(p) for p in out]


def asset_dir() -> str:
    for p in _candidates():
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(
        "找不到 wheelbipe 的 USD 资产目录。请设置环境变量 WHEELBIPE_ASSET_ROOT 指向 "
        "wheeled-legged_RL/source/agent_world/agent_world/assets/usd_files"
    )


def usd_path(no_gimbal: bool = False) -> str:
    """Compatibility name retained for callers, now returning the MJCF asset."""
    return mjcf_path()


# MJLab robot entity configuration. Actuator gains and limits stay in the
# converted MJCF so sim2sim and training use the same actuator model.
def mjcf_path() -> str:
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "assets", "mujoco", "wheelbipe_v14_2.xml")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"找不到 MJCF: {path}。先运行 `uv run python tools/usd2mjcf.py --fk-json`。"
        )
    return path


def make_robot_cfg(usd: str | None = None, prim_path: str | None = None):
    """Build an MJLab entity from the converted wheelbipe MJCF."""
    import mujoco
    from mjlab.actuator import XmlActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

    source = usd if usd and usd.endswith(".xml") else mjcf_path()
    actuators = (
        XmlActuatorCfg(
            target_names_expr=(".*_front1_joint", ".*_rear1_joint", "gimbal_.*_joint"),
            command_field="position",
        ),
        XmlActuatorCfg(
            target_names_expr=(".*_wheel_joint",),
            command_field="velocity",
        ),
        XmlActuatorCfg(
            target_names_expr=(".*_spring2_joint",),
            command_field="effort",
        ),
    )
    def load_entity_spec():
        spec = mujoco.MjSpec.from_file(source)
        spec.option.timestep = 0.002
        spec.option.tolerance = 1e-8
        spec.option.impratio = 1.0
        spec.option.integrator = int(mujoco.mjtIntegrator.mjINT_EULER)
        spec.option.cone = int(mujoco.mjtCone.mjCONE_PYRAMIDAL)
        return spec

    return EntityCfg(
        spec_fn=load_entity_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.38),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(actuators=actuators),
    )


# 关节分组
# 实现在 spec.py（不依赖 isaaclab），这里只做 re-export
from .spec import RobotSpec, build_spec  # noqa: E402,F401

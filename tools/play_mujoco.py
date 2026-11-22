# SPDX-License-Identifier: MIT
"""在 MuJoCo（纯 CPU）里跑 SuperMimic 训练出来的策略，键盘遥控轮腿机器人。"""

from __future__ import annotations

import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from supermimic_wheelbipe import networks  # noqa: E402
from supermimic_wheelbipe.constants import ACTION_CLIP, default_term_height  # noqa: E402
from supermimic_wheelbipe.command import CommandLib  # noqa: E402
from supermimic_wheelbipe.gamepad import BUTTONS as PAD_BUTTONS  # noqa: E402
from supermimic_wheelbipe.gamepad import GamepadCmd, XInputPad, describe as pad_desc  # noqa: E402
from supermimic_wheelbipe.gamepad import selftest as pad_selftest  # noqa: E402
from supermimic_wheelbipe.mdp import MDP  # noqa: E402
from supermimic_wheelbipe.mujoco_env import DEFAULT_XML, MujocoWheelbipe, load_policy_ckpt  # noqa: E402

VX_LIM = (-1.5, 1.5)
WZ_LIM = (-10.21, 10.21)
DEFAULT_HEIGHT_LIMITS = (0.16, 0.28)
VX_RAMP, WZ_RAMP, H_RAMP = 0.02, 0.06, 0.002      # 每个控制步（50 Hz）→ 1 m/s²、3 rad/s²、0.1 m/s
VX_SLEW, WZ_SLEW = 0.04, 0.2                      # 下发指令的变化率上限（2 m/s²、10 rad/s²）
TERM_COS = float(np.cos(np.deg2rad(40.0)))

class KeyboardCmd:
    """仅在独立的驾驶面板获得焦点时接收键盘事件。"""

    KEY_ALIASES = {"up": "w", "down": "s", "left": "a", "right": "d"}
    DRIVE_KEYS = frozenset("wasdqe")

    def __init__(self, vx, wz, height, height_limits):
        self.vx, self.wz, self.h = vx, wz, height
        self.height_limits = height_limits
        self.quit = False
        self.reset = False
        self._pressed = set()
        self._window = None

    def open_panel(self):
        import tkinter as tk
        from tkinter import ttk

        window = tk.Tk()
        self._window = window
        window.title("Wheelbipe 驾驶面板")
        window.resizable(False, False)
        window.geometry("+30+60")
        content = ttk.Frame(window, padding=16)
        content.grid(sticky="nsew")
        self._focus_status = tk.StringVar(value="点击此窗口接管驾驶键盘")
        self._telemetry = tk.StringVar()
        ttk.Label(content, textvariable=self._focus_status).grid(sticky="w", pady=(0, 10))
        ttk.Label(
            content,
            text="W / S 或 ↑ / ↓   增减前向速度        A / D 或 ← / →   增减转速\n"
                 "Q / E   升降目标高度        1 / 2 / 3   0.3 / 0.8 / 1.4 m/s\n"
                 "空格   目标速度归零        R   复位        Esc   退出",
            justify="left",
        ).grid(sticky="w")
        ttk.Label(content, textvariable=self._telemetry, justify="left").grid(sticky="w", pady=(12, 10))
        buttons = ttk.Frame(content)
        buttons.grid(sticky="w")
        ttk.Button(buttons, text="停车", command=self.stop_motion, takefocus=False).pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="复位", command=self.request_reset, takefocus=False).pack(side="left")
        window.bind("<KeyPress>", self._on_press)
        window.bind("<KeyRelease>", self._on_release)
        window.bind("<FocusIn>", self._on_focus_in)
        window.bind("<FocusOut>", self._on_focus_out)
        window.protocol("WM_DELETE_WINDOW", self.request_quit)
        window.update()
        window.lift()
        window.focus_force()

    def stop_motion(self):
        self._pressed.clear()
        self.vx = self.wz = 0.0

    def request_reset(self):
        self.stop_motion()
        self.reset = True

    def request_quit(self):
        self.quit = True

    def _key(self, event):
        name = event.keysym.lower()
        return self.KEY_ALIASES.get(name, name)

    def _on_press(self, event):
        key = self._key(event)
        if key in self.DRIVE_KEYS:
            self._pressed.add(key)
        elif key == "space":
            self.stop_motion()
        elif key in ("1", "2", "3"):
            self.vx = {"1": 0.3, "2": 0.8, "3": 1.4}[key]
        elif key == "r":
            self.request_reset()
        elif key == "escape":
            self.request_quit()
        else:
            return None
        return "break"

    def _on_release(self, event):
        self._pressed.discard(self._key(event))

    def _on_focus_in(self, event):
        if event.widget is self._window:
            self._focus_status.set("键盘控制已启用")

    def _on_focus_out(self, event):
        if event.widget is self._window:
            self.stop_motion()
            self._focus_status.set("键盘已停用 · 点击此窗口恢复驾驶")

    def show_telemetry(self, vx, wz, height):
        if self._window is not None:
            self._telemetry.set(
                f"前向 {vx:+.2f} / {self.vx:+.2f} m/s     偏航 {wz:+.2f} / {self.wz:+.2f} rad/s\n"
                f"高度 {height:.3f} / {self.h:.3f} m     （实测 / 目标）"
            )

    def update(self):
        """每个控制步调用：按住即按斜坡改目标。"""
        if self._window is None:
            return
        self._window.update()
        p = self._pressed
        if "w" in p:
            self.vx += VX_RAMP
        if "s" in p:
            self.vx -= VX_RAMP
        if "a" in p:
            self.wz += WZ_RAMP
        if "d" in p:
            self.wz -= WZ_RAMP
        if "q" in p:
            self.h += H_RAMP
        if "e" in p:
            self.h -= H_RAMP
        self.vx = float(np.clip(self.vx, *VX_LIM))
        self.wz = float(np.clip(self.wz, *WZ_LIM))
        self.h = float(np.clip(self.h, *self.height_limits))

    def stop(self):
        if self._window is not None:
            self._window.destroy()
            self._window = None


def fmt_cmd(c):
    return f"vx={c.vx:+.2f} wz={c.wz:+.2f} h={c.h:.3f}"


def main():
    ap = argparse.ArgumentParser(description="MuJoCo 里用键盘遥控 SuperMimic 轮腿策略")
    ap.add_argument("--ckpt", type=str,
                    default=os.path.join("logs", "wb_final", "last.pt"),
                    help="SuperMimic 检查点；--teacher 时只用来读 spec")
    ap.add_argument("--xml", type=str, default=DEFAULT_XML)
    ap.add_argument("--cmd", type=float, nargs=3, default=None, metavar=("VX", "WZ", "H"),
                    help="初始指令；省略时原地站立，高度取训练区间中点")
    ap.add_argument("--steps", type=int, default=0, help="headless 模式跑多少控制步（0=一直跑）")
    ap.add_argument("--headless", action="store_true", help="不开窗口（无键盘，按 --cmd 恒定指令）")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--camera", type=str, default="track",
                    help="track（跟随）/ fixed / free")
    ap.add_argument("--trim", type=float, default=0.0,
                    help="外环速度修正增益（1/s，0=关闭）。策略看不到机身速度，换引擎后稳态会偏；"
                         "这个慢积分把实测速度补回指令上（限幅 ±0.4 m/s）")
    ap.add_argument("--gamepad", type=str, default="auto", choices=["auto", "off"],
                    help="auto=插上 XInput 手柄就接管 vx/wz；off=只认键盘")
    ap.add_argument("--pad_deadzone", type=float, default=0.08, help="摇杆死区（0~1）")
    ap.add_argument("--pad_expo", type=float, default=0.0,
                    help="摇杆曲线指数偏置：0=纯线性（默认），0.5=中间更细腻")
    ap.add_argument("--wz_ramp", type=float, default=0.0,
                    help="右杆左右 -> 偏航**爬升率**（rad/s²）。0=直接线性给转速；"
                         "设成 12 就是'推住杆、小陀螺线性起来'")
    ap.add_argument("--pad_probe", type=float, default=0.0,
                    help="只测手柄：打印 N 秒实时杆值（没插手柄时跑合成自检）后退出")
    ap.add_argument("--teacher", type=str, default=None,
                    help="用预训练 PPO 代替 SuperMimic 策略（auto 或 run 目录）。"
                         "它在 MuJoCo 里跟踪误差只有 0.03~0.04 m/s，比 wb_final 准得多")
    a = ap.parse_args()

    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if a.device == "cpu":
        torch.set_num_threads(1)
    networks.AMP_DTYPE = None

    # 手柄自检（不加载模型，插不上也能看响应曲线）
    if a.pad_probe > 0:
        pad = XInputPad()
        print(pad_desc(pad))
        if not pad.ok:
            return 1
        if pad.poll() is None:
            print("没检测到手柄 —— 下面用合成杆值验证映射曲线：")
            pad_selftest()
            print("\n接上手柄后（XInput 模式：Xbox 系 / 8BitDo 的 X 档 / PS 手柄经 DS4Windows）"
                  "再跑一次 --pad_probe 5 就能看到实时杆值。")
            return 0
        print(f"读手柄 {a.pad_probe:.0f}s（动一动杆和按键）：")
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < a.pad_probe:
            st = pad.poll()
            if st is None:
                print("  手柄断开")
                break
            held = ",".join(k for k, v in PAD_BUTTONS.items() if st["buttons"] & v)
            print(f"  lx={st['lx']:+.2f} ly={st['ly']:+.2f} rx={st['rx']:+.2f} "
                  f"ry={st['ry']:+.2f} lt={st['lt']:.2f} rt={st['rt']:.2f}  {held}")
            time.sleep(0.1)
        return 0

    ck = a.ckpt if os.path.isabs(a.ckpt) else os.path.normpath(os.path.join(REPO, a.ckpt))
    pi, spec, args = load_policy_ckpt(ck, a.device)   # 即使 --teacher 也要它来拿 spec
    K, Hh = int(args["goal_frames"]), int(args["hist"])
    env = MujocoWheelbipe(spec, xml=a.xml)

    teacher = None
    if a.teacher:
        from supermimic_wheelbipe.teacher import Teacher, pick_best_teacher

        tck = pick_best_teacher(a.teacher)
        teacher = Teacher(spec, spec.joint_names, torch.zeros(spec.nj), tck, device=a.device)
        print(f"[teacher] {tck}")
    else:
        print(f"[ckpt] {os.path.basename(ck)}  it={args['iteration']} "
              f"obs={spec.nq + spec.nd + 6 + spec.na} goal={4 * K} hist={Hh}")
    print(env.describe())

    z_contact, z_stand = env.settle_stand_height()
    term_h = args.get("term_height")
    if term_h is None:
        term_h = default_term_height(z_stand)
    print(f"[stand] FK 触地 {z_contact:.4f} m / 静置 {z_stand:.4f} m，终止高度 {term_h:.4f} m")

    height_limits = tuple(args.get("height_range") or DEFAULT_HEIGHT_LIMITS)
    vx, wz, height = a.cmd if a.cmd is not None else (0.0, 0.0, sum(height_limits) / 2)
    cmd = CommandLib.fixed([[vx, 0.0, wz, height]], device=a.device, frames=K)
    mdp = MDP(spec, cmd, kin=None, default_q=torch.zeros(spec.nj), qd_limit=None)
    kb = KeyboardCmd(vx, wz, height, height_limits)
    pad = None
    if a.gamepad != "off":
        pad = XInputPad()
        print(pad_desc(pad))
    keys = GamepadCmd(kb, pad, enabled=(not a.headless and pad is not None and pad.ok),
                      vx_max=VX_LIM[1], wz_max=WZ_LIM[1], h_rate=H_RAMP * 50 * 1.2,
                      h_lo=height_limits[0], h_hi=height_limits[1], deadzone=a.pad_deadzone,
                      expo=a.pad_expo, wz_ramp=a.wz_ramp)
    print(f"[start] {fmt_cmd(keys)}" + ("   [手柄已接管]" if keys.connected else ""))

    viewer = None
    if not a.headless:
        try:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(env.m, env.d)
            if a.camera == "track":
                viewer.cam.type = 1  # mjCAMERA_TRACKING
                viewer.cam.trackbodyid = env.base_id
                viewer.cam.distance = 1.6
                viewer.cam.azimuth = 120.0
                viewer.cam.elevation = -18.0
            kb.open_panel()
        except Exception as e:  # noqa: BLE001
            keys.stop()
            if viewer is not None:
                viewer.close()
            raise RuntimeError("无法打开 MuJoCo 或驾驶面板窗口") from e

    # 状态
    env.reset(z=z_stand)
    hist = None
    a_prev = torch.zeros(1, spec.na, device=a.device)
    t_alive = 0.0
    n_reset = 0
    t0 = time.perf_counter()
    n = 0
    T = env.control_dt
    next_t = t0
    trim = 0.0          # 外环速度修正量（见 --trim）
    vx_cmd = 0.0        # 经过变化率限幅后真正下发的指令
    wz_cmd = 0.0

    def do_reset(msg=None):
        nonlocal hist, a_prev, t_alive, n_reset, trim, vx_cmd, wz_cmd
        env.reset(q_full=None, z=z_stand)
        hist, a_prev, t_alive, trim = None, torch.zeros(1, spec.na, device=a.device), 0.0, 0.0
        vx_cmd = wz_cmd = 0.0        # 复位后从静止平滑起步，别给策略一个阶跃
        n_reset += 1
        if teacher is not None:
            teacher.reset()
        if msg:
            print(msg)

    try:
        while viewer is None or viewer.is_running():
            keys.update(T)
            if keys.quit:
                break
            if keys.reset:
                keys.reset = False
                do_reset("[reset] 手动重置")

            # 指令变化率限幅 2 m/s²、10 rad/s²，阶跃指令会让俯仰冲到 50° 直接翻
            vx_t = float(np.clip(keys.vx + trim, -1.6, 1.6))
            vx_cmd += max(-VX_SLEW, min(VX_SLEW, vx_t - vx_cmd))
            wz_cmd += max(-WZ_SLEW, min(WZ_SLEW, keys.wz - wz_cmd))
            cmd.cmd[:] = cmd.cmd.new_tensor((vx_cmd, 0.0, wz_cmd, keys.h))

            S = env.state()
            if teacher is not None:
                with torch.no_grad():
                    act = teacher.act(S, cmd.cmd[:1]).clamp(-ACTION_CLIP, ACTION_CLIP)
            else:
                o = mdp.obs(S, a_prev)
                hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat(
                    (hist[:, 1:], o.unsqueeze(1)), 1)
                with torch.no_grad():
                    act = pi(hist, cmd.cmd.reshape(1, -1)).clamp(-ACTION_CLIP, ACTION_CLIP)
            env.step(act)
            a_prev = act
            t_alive += T
            n += 1

            # 外环速度修正：策略看不到机身速度（obs 里只有轮速），引擎一换稳态就会偏
            # （实测 h=0.24 时恒定 -0.24 m/s，松手还会漂）。这里用一个慢积分补偿指令。
            v_meas = env.vel_forward
            if a.trim > 0:
                trim = float(np.clip(trim + a.trim * (keys.vx - v_meas) * T, -0.4, 0.4))

            # 终止判定（与 mdp.termination 同口径）
            z, cos_t = env.base_z, env.tilt_cos
            dead = (z < term_h) or (cos_t < TERM_COS) or not np.isfinite(z)
            if dead:
                do_reset(f"[term] 摔了：z={z:.3f} cos={cos_t:.3f}，自动重置")

            wz = float(env.d.qvel[5])
            if n % max(1, a.log_every) == 0:
                print(f"  t={t_alive:5.2f}s z={z:.3f} cos={cos_t:.3f} | "
                      f"cmd {fmt_cmd(keys)} | v={v_meas:+.2f}/{keys.vx:+.2f} "
                      f"w={wz:+.2f}/{keys.wz:+.2f} trim={trim:+.2f}")

            if viewer is not None:
                kb.show_telemetry(v_meas, wz, z)
                try:  # 3.x 有 set_texts，老版本没有就忽略
                    viewer.set_texts((f"z={z:.3f}  v={v_meas:+.2f}/{keys.vx:+.2f}  "
                                      f"wz={wz:+.2f}/{keys.wz:+.2f}  h={keys.h:.2f}",))
                except Exception:  # noqa: BLE001
                    pass
                viewer.sync()

            # 实时节拍（严格 50 Hz，落后就全速追）
            next_t += T
            dt = next_t - time.perf_counter()
            if viewer is not None and dt > 0:
                time.sleep(dt)

            if a.steps and n >= a.steps:
                break
    except KeyboardInterrupt:
        print("\n[exit] Ctrl+C")
    finally:
        keys.stop()
        if viewer is not None:
            try:
                viewer.close()
            except Exception:  # noqa: BLE001
                pass

    wall = time.perf_counter() - t0
    print(f"\n[done] {n} 控制步 / {n * T:.2f}s 仿真 / {wall:.1f}s 墙钟 "
          f"({n * T / max(wall, 1e-6):.1f}x 实时)  重置 {n_reset} 次")
    return 0


if __name__ == "__main__":
    sys.exit(main())

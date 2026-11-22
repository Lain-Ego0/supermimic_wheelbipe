# 安装与运行

## 环境

项目使用 Python 3.12、MJLab 1.6.0、MuJoCo/MuJoCo Warp 3.11 和 CUDA PyTorch。依赖及兼容的 `mjviser` 版本由 `uv.lock` 固定。

```bash
uv python install 3.12
uv sync --locked
uv run python -c "import mujoco, mjlab, torch; print(mujoco.__version__, torch.cuda.is_available())"
```

`uv sync --locked` 在项目根目录创建 `.venv`。训练与评估通过 MJLab `Scene`、`Entity` 和 Warp `Simulation` 运行，物理步长为 0.0005 秒，策略控制周期为 0.02 秒。MJCF 中额外的闭链导向关节仍参与仿真，接口按已有检查点的关节名称和顺序组装观察量。

## 标定

```bash
uv run python tools/calibrate.py --device cuda:0
```

标定会打印 MJLab 关节清单、动作分组、弹簧模型、MJCF 前向运动学误差和站立高度。模型刚体位置的 FK 校验误差应低于 0.01 米。可用 `--settle_steps` 调整零动作静置步数，已知站立高度时可以用 `--stretch` 覆盖。

## 训练

```bash
uv run python supermimic_wheelbipe/train.py --device cuda:0 --headless \
  --num_envs 4096 --iterations 24000 --wm_only_iters 800 --warmup_steps 1500 \
  --log_every 10 --horizon 16 --compile 0 --utd 2.0 \
  --wm_batches 5 --wm_batch 8192 --pi_batch 4096 \
  --lin_vel_x -1.5 1.5 --ang_vel_z -10.21 10.21 --height_range 0.20 0.28 \
  --log_dir logs/wb_final
```

使用 `--resume logs/wb_final/last.pt` 从完整训练检查点续训。单独的推理策略检查点可用于评估和回放，但不包含世界模型及优化器状态。`run_train.sh` 会调用项目 `.venv` 中的 Python。

零动作静置默认执行 120 个策略步，也就是 2.4 秒。`--z_stand` 可以指定已知站立高度，`--settle_steps 0` 可跳过静置。高度指令默认取站立高度附近的窗口，可由 `--height_range` 和 `--term_height` 指定。

训练日志保存在 `logs/<run>/train_log.csv`。`wm_loss` 和 `wm_err8` 表示世界模型误差，`imag_survival` 表示想象 rollout 存活率，`sim_rew` 表示仿真单步奖励，`ep_len`、`ep_ret_norm` 和 `term_frac` 表示真实仿真回合表现。

## 评估

```bash
uv run python supermimic_wheelbipe/evaluate.py \
  --device cuda:0 --ckpt logs/wb_final/ckpt_024000.pt
```

默认评估 16 条固定速度和偏航指令。使用 `--run_dir logs/wb_final --watch` 持续评估新检查点。旧 Isaac Lab 检查点保留原关节名和观察顺序，MJLab 实体按名称映射关节状态及执行器目标。

## 交互回放

```bash
uv run python tools/play.py --ckpt logs/wb_final/ckpt_024000.pt --gamepad off
```

回放打开 MuJoCo 窗口和独立的「Wheelbipe 驾驶面板」。W/S 或方向键调节前向速度，A/D 或方向键调节偏航，Q/E 调节目标高度。空格将速度目标归零，R 复位，Esc 退出。键盘驾驶只在面板获得焦点时生效，MuJoCo 窗口继续接收相机快捷键。该入口使用与 MJLab 训练相同的 MJCF 和策略检查点。

不打开窗口时可以执行有限步数的回放：

```bash
uv run python tools/play.py --ckpt logs/wb_final/ckpt_024000.pt \
  --headless --steps 400 --cmd 0.8 0.0 0.24 --gamepad off
```

## 工具

| 脚本 | 用途 |
|---|---|
| `tools/selftest_cpu.py` | 网络、缓冲区、命令库和奖励逻辑自检 |
| `tools/calibrate.py` | MJLab 实体、MJCF 运动学和站立状态标定 |
| `tools/run_teacher.py` | 教师策略采样、回放和行为克隆 |
| `supermimic_wheelbipe/train.py` | SuperMimic 训练主循环 |
| `supermimic_wheelbipe/evaluate.py` | MJLab GPU 固定指令评估 |
| `tools/play.py` | MuJoCo GUI 策略回放和驾驶面板 |
| `tools/evaluate_mujoco.py` | CPU MuJoCo 跨仿真评估 |

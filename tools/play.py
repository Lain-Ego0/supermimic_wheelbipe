#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Interactive MuJoCo replay for policies trained with the MJLab backend.

The focused Tk control panel lives in ``play_mujoco.py``. It receives driving
keys only while the panel has focus, so the MuJoCo viewer keeps its camera
shortcuts available.
"""

from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "tools", "play_mujoco.py")

if __name__ == "__main__":
    forwarded = list(sys.argv[1:])
    if "--gamepad" not in forwarded:
        forwarded.extend(("--gamepad", "off"))
    os.execv(sys.executable, [sys.executable, SCRIPT, *forwarded])

"""Launch the website development server and scaling-data watcher."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def website() -> None:
    root = Path(__file__).resolve().parents[2]
    command = ["pnpm", "--dir", str(root / "website"), "run", "dev"]
    command.extend(sys.argv[1:])
    os.execvp(command[0], command)

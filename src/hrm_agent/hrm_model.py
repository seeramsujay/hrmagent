"""The official Sapient HRM (sapientinc/HRM), used the way it was built: grid in, grid out.

HRM is not a language model. It cannot read a problem statement or write code; it maps a
fixed-size token grid to a solved grid, and each checkpoint only solves the task it was trained
on. This module runs the official 30x30 maze checkpoint through the loader in the sibling
PALAIS repository (`palais_hrm`), so there is a single HRM implementation in the monorepo.

Set PALAIS_ROOT if PALAIS is not checked out next to this repository.
"""

import os
import sys
from pathlib import Path
from typing import Any

PALAIS_ROOT = Path(os.environ.get("PALAIS_ROOT", Path(__file__).resolve().parents[3] / "PALAIS"))


def solve_maze(maze: str, device: str = "auto") -> dict[str, Any]:
    """Solve a 30x30 maze: 30 lines using '#' wall, ' ' free, 'S' start, 'G' goal."""
    if not (PALAIS_ROOT / "palais_hrm").is_dir():
        raise FileNotFoundError(f"palais_hrm not found under {PALAIS_ROOT}; set PALAIS_ROOT.")
    if str(PALAIS_ROOT) not in sys.path:
        sys.path.insert(0, str(PALAIS_ROOT))
    from palais_hrm.maze import MazeSolver

    return MazeSolver.shared(None if device == "auto" else device).solve_text(maze)

"""Build the SUMO intersection network.

Usage
-----
    .\\.venv\\Scripts\\python.exe scripts\\sumo_build.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sumo.network import build_intersection  # noqa: E402


def main() -> int:
    cfg = build_intersection()
    print(f"net : {cfg.parent / 'intersection.net.xml'}")
    print(f"cfg : {cfg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

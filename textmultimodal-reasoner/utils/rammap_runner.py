"""Helpers to invoke an external RamMap cleanup script."""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path

RAMMAP_SCRIPT_RELATIVE = Path("tools/rammap/run_rammap_clean.vbs")


def run_rammap_cleanup(project_root: Path) -> bool:
    """Run the RamMap cleanup VBS script, if present, on Windows."""
    if platform.system().lower() != "windows":
        return False

    script_path = project_root / RAMMAP_SCRIPT_RELATIVE
    if not script_path.exists():
        return False

    result = subprocess.run(
        ["wscript.exe", str(script_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0

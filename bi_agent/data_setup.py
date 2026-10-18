"""Make sure the synthetic database exists and matches this version of the code.

Used by the app on startup (Streamlit Cloud has no setup step) and safe to call repeatedly.
A database from an older version (no planted-truth table) is rebuilt automatically.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from . import config, db


def is_current(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        db.connect(str(path)).execute("SELECT 1 FROM _ground_truth")
        return True
    except Exception:  # noqa: BLE001
        return False


def ensure_current_data(path: str | Path | None = None) -> bool:
    """Returns True if the database had to be (re)built."""
    p = Path(path or config.DB_PATH)
    if is_current(p):
        return False
    sys.path.insert(0, str(config.ROOT / "scripts"))
    import generate_data  # noqa: E402

    db.connect.cache_clear()
    db._VALUES_CACHE.clear()
    shutil.rmtree(p.parent / "ds_cache", ignore_errors=True)
    generate_data.write_duckdb(generate_data.build(), str(p))
    db.connect.cache_clear()
    return True

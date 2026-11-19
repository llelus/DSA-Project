"""Single place where notebooks write figures and headline numbers.

README numbers must come from results/metrics.json, never be typed by hand —
v1's README drifted away from the notebook outputs several times.
"""
import json

import numpy as np

from . import config


def _plain(v):
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        # 6 significant digits, not 6 decimals: p-values like 1e-22 must not become 0
        return None if np.isnan(v) else float(f"{float(v):.6g}")
    return v


def save_metrics(section, **values):
    """Merge values into results/metrics.json under `section`."""
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    data = json.loads(config.METRICS.read_text(encoding="utf-8")) if config.METRICS.exists() else {}
    data.setdefault(section, {}).update(_plain(values))
    config.METRICS.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False),
                              encoding="utf-8")
    print(f"saved {len(values)} value(s) to results/metrics.json [{section}]")


def load_metrics():
    return json.loads(config.METRICS.read_text(encoding="utf-8")) if config.METRICS.exists() else {}


def savefig(fig, name):
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path

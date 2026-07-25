#!/usr/bin/env python3
"""Smoke test: two takeoff sample runs with the same seed match milestone dates."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import forecasting_takeoff as ft  # noqa: E402


def _sar_dates(config: dict, n_sims: int) -> list:
    samples = ft.get_milestone_samples(config, n_sims)
    out = []
    for i in range(n_sims):
        dates = ft.run_single_simulation(samples, i)
        if dates:
            out.append(dates[0].isoformat())
    return out


def main() -> None:
    with open(ROOT / "params.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["simulation"]["n_sims"] = 32
    cfg["simulation"]["seed"] = 20260725
    a = _sar_dates(cfg, cfg["simulation"]["n_sims"])
    b = _sar_dates(cfg, cfg["simulation"]["n_sims"])
    if a != b:
        raise SystemExit(f"seed smoke failed: {a[:3]} != {b[:3]}")
    print(f"seed smoke ok: n={len(a)} identical SAR dates")


if __name__ == "__main__":
    main()

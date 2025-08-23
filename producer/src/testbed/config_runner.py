import argparse
import yaml

from pathlib import Path

from . import main as old_cli
from .file_bencher import Metrics

def main():
    parser = argparse.ArgumentParser(description="YAML-driven Draco benchmark runner (uses existing CLI module)")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with config_path.open("r") as fh:
        cfg = yaml.safe_load(fh) or {}

    base_dir = config_path.parent

    defaults = cfg.get("defaults", {}) or {}
    experiments = cfg.get("experiments", []) or []
    output = cfg.get("output", {}) or {}
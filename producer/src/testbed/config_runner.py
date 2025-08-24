import argparse
import csv
import os
import yaml

from pathlib import Path
from typing import List, Dict, Union

from .yaml_config_reader import YamlReader
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

    # e.g. "results/{name}.csv"
    per_experiment_csv = output.get("per_experiment_csv", "")
    # e.g. "results/all.csv"
    combined_csv = output.get("combined_csv", "")

    all_rows = []

    print(f"Received experiments to process:\n{experiments}")

    for experiment in experiments:

        experiment_name = experiment["name"]
        experiment_inputs = experiment["inputs"]

        single_draco = experiment.get("draco", {}) or {}
        grid = experiment.get("grid", {}) or {}

        repeats = YamlReader.effective_int(
            defaults=defaults, 
            overrides=experiment, 
            key="repeats", 
            fallback=3
        )

        warmup = YamlReader.effective_bool(
            defaults=defaults, 
            overrides=experiment, 
            key="warmup", 
            fallback=False
        )

        drop_first = YamlReader.effective_bool(
            defaults=defaults, 
            overrides=experiment, 
            key="drop_first", 
            fallback=False
        )

        resolved_inputs = []
        for pat in experiment_inputs:
            p = Path(pat)
            if not p.is_absolute():
                resolved_inputs.append(str((base_dir / p).as_posix()))
            else:
                resolved_inputs.append(str(p))
        
        grid_items = list(YamlReader.iter_grid(grid=grid))

        for index, grid_parameters in enumerate(grid_items):

            draco_config = YamlReader.merge_draco(
                defaults=defaults,
                single=single_draco,
                grid_params=grid_parameters
            );

            print(
                f"Config {index}/{len(grid_items)}| name {experiment_name} "
                f"| qpos={draco_config['qpos']} | qcol={draco_config['qcol']} "
                f"| speed_encode={draco_config['speed_encode']} | speed_decode={draco_config['speed_decode']} "
                f"| (repeats={repeats}, warmup={warmup}, drop_first={drop_first})"
            )

            rows = old_cli.run(
                inputs=resolved_inputs,
                repeats=repeats,
                qpos=draco_config["qpos"],
                qcol=draco_config["qcol"],
                speed_encode=draco_config["speed_encode"],
                speed_decode=draco_config["speed_decode"],
                warmup=warmup,
                drop_first=drop_first,
            )

            print(f"Got result from run:\n{rows}")

            extra_cols = {
                "experiment": experiment_name,
                "qpos": draco_config["qpos"],
                "qcol": draco_config["qcol"],
                "speed_encode":  draco_config["speed_encode"],
                "speed_decode": draco_config["speed_decode"],
                "repeats": repeats,
                "warmup": warmup,
                "drop_first": drop_first,
            }

            for row in rows:
                rec = row.as_row()
                rec.update(extra_cols)
                all_rows.append(rec)

            if per_experiment_csv:
                path = per_experiment_csv.format(name=experiment_name)
                csv_path = (base_dir / path) if not os.path.isabs(path) else Path(path)
                YamlReader.rows_to_csv(rows=rows, extra_cols=extra_cols, csv_path=csv_path)

    if combined_csv and all_rows:
        combined_path = (base_dir / combined_csv) if not os.path.isabs(combined_csv) else Path(combined_csv)
        YamlReader.ensure_parent(combined_path)
        fieldnames = list(all_rows[0].keys())
        with combined_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)
        print(f"[ALL] wrote -> {combined_path}") 

if __name__ == "__main__":
    main()
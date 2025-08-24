import csv
import itertools

from pathlib import Path
from typing import Dict, List, Union, Iterable

from .file_bencher import Metrics

class YamlReader:
    def __init__(self):
        raise NotImplementedError("")

    @staticmethod
    def effective_bool(defaults: dict, overrides: dict, key: str, fallback: bool) -> bool:
        return bool(overrides.get(key, defaults.get(key, fallback)))

    @staticmethod
    def effective_int(defaults: dict, overrides: dict, key: str, fallback: int) -> int:
        return int(overrides.get(key, defaults.get(key, fallback)))

    @staticmethod
    def ensure_parent(path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def merge_draco(defaults: dict, single: dict, grid_params: dict) -> Dict[str, Union[int, float]]: 
        """Merge order: defaults.draco -> single draco -> grid params.""" 
        d = (defaults.get("draco") or {}).copy() 
        d.update(single or {}) 
        d.update(grid_params or {}) 
        return { 
            "qpos": int(d.get("qpos", 11)), 
            "qcol": int(d.get("qcol", 8)), 
            "speed_encode": int(d.get("speed_encode", 10)), 
            "speed_decode": int(d.get("speed_decode", 10)), 
        }

    @staticmethod
    def iter_grid(grid: Dict[str, List[Union[int, float]]]) -> Iterable[Dict[str, Union[int, float]]]:
        """
        Cartesian product of grid params. Returns at least one empty dict if no grid.
        """
        if not grid:
            yield {}
            return
        keys = list(grid.keys())
        for vals in itertools.product(*(grid[k] for k in keys)):
            yield dict(zip(keys, vals))

    @staticmethod
    def ensure_parent(path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def rows_to_csv(rows: List[Metrics], extra_cols: Dict[str, Union[str, int, float]], csv_path: Path):

        if not rows:
            return

        YamlReader.ensure_parent(csv_path)
        # base fieldnames from Metrics
        base = list(rows[0].as_row().keys())
        extra = list(extra_cols.keys())
        fieldnames = base + [c for c in extra if c not in base]
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for m in rows:
                row = m.as_row()
                row.update(extra_cols)
                w.writerow(row)




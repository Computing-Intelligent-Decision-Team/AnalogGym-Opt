import os
from typing import Any, Dict, List

import yaml


class CircuitConfigLoader:
    """Load circuit YAML files and resolve their simulation paths."""

    def __init__(self, config_dir: str = "circuit_configs", load_all: bool = False):
        self.config_dir = os.path.abspath(config_dir)
        self._circuit_configs: Dict[str, Dict[str, Any]] = {}
        self._available_circuits = None

        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
            self._available_circuits = []

        if load_all:
            self._load_all_configs()
        else:
            self._scan_available_circuits()

    def _scan_available_circuits(self) -> None:
        """Record available circuit names from YAML files."""
        self._available_circuits = []
        for filename in os.listdir(self.config_dir):
            if filename.endswith((".yaml", ".yml")):
                self._available_circuits.append(os.path.splitext(filename)[0])

    def _load_all_configs(self) -> None:
        """Load all available circuit configs."""
        self._scan_available_circuits()
        for circuit_name in self._available_circuits:
            try:
                self._load_single_config(circuit_name)
            except Exception as exc:
                print(f"Failed to load {circuit_name}.yaml: {exc}")

    def _load_single_config(self, circuit_name: str) -> Dict[str, Any]:
        """Load one circuit config by name."""
        if circuit_name in self._circuit_configs:
            return self._circuit_configs[circuit_name]

        yaml_path = os.path.join(self.config_dir, f"{circuit_name}.yaml")
        yml_path = os.path.join(self.config_dir, f"{circuit_name}.yml")

        if os.path.exists(yaml_path):
            config_path = yaml_path
        elif os.path.exists(yml_path):
            config_path = yml_path
        else:
            if self._available_circuits is not None and circuit_name not in self._available_circuits:
                raise ValueError(
                    f"Unknown circuit '{circuit_name}'. Available circuits: {self.get_available_circuits()}"
                )
            raise ValueError(f"Circuit config not found: {circuit_name}.yaml or {circuit_name}.yml")

        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        self._validate_config(config, circuit_name)
        config["paths"] = self._prepare_paths(config)
        self._normalize_performance_weights(config)
        config["config_path"] = config_path
        config["name"] = circuit_name
        config["category"] = circuit_name.split("_")[0] if "_" in circuit_name else "default"

        self._circuit_configs[circuit_name] = config
        return config

    def _validate_config(self, config: Dict[str, Any], circuit_name: str) -> None:
        """Check required top-level fields."""
        for field in ("base_dir", "device", "performance", "file_paths"):
            if field not in config:
                raise ValueError(f"Circuit '{circuit_name}' is missing required field: {field}")

    def get_available_circuits(self) -> List[str]:
        """Return all available circuit names."""
        if self._available_circuits is None:
            self._scan_available_circuits()
        return self._available_circuits.copy()

    def _convert_str_numbers(self, obj):
        """Recursively convert numeric strings to numbers."""
        if isinstance(obj, dict):
            return {key: self._convert_str_numbers(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [self._convert_str_numbers(item) for item in obj]
        if isinstance(obj, str):
            try:
                value = float(obj)
                return int(value) if value.is_integer() else value
            except ValueError:
                return obj
        return obj

    def get_circuit_config(self, circuit_name: str) -> Dict[str, Any]:
        """Return a loaded circuit config with numeric strings converted."""
        config = self._load_single_config(circuit_name)
        return self._convert_str_numbers(config.copy())

    def get_simulation_dir(self, circuit_name: str) -> str:
        """Return the configured simulation template directory."""
        config = self.get_circuit_config(circuit_name)
        return config.get("base_dir", "")

    def _prepare_paths(self, config):
        """Resolve file paths relative to the repository root and base_dir."""
        project_root = os.path.dirname(os.path.abspath(__file__))
        base_dir = config.get("base_dir", "")
        file_paths = config["file_paths"]
        paths = {}

        for key, rel_path in file_paths.items():
            if base_dir:
                normalized_base = base_dir[3:] if base_dir.startswith("../") else base_dir
                full_path = os.path.join(project_root, normalized_base, rel_path)
            else:
                full_path = os.path.join(project_root, rel_path)
            paths[key] = os.path.normpath(full_path)

        return paths

    def _normalize_performance_weights(self, config: Dict[str, Any]) -> None:
        """Normalize explicit performance weights so they sum to one."""
        performance = config.get("performance", {})
        total_weight = sum(
            float(value.get("weight", 0.0))
            for value in performance.values()
            if isinstance(value, dict) and "weight" in value
        )

        if total_weight > 0:
            for value in performance.values():
                if isinstance(value, dict) and "weight" in value:
                    value["weight"] = float(value["weight"]) / total_weight

    def reload_configs(self) -> None:
        """Clear the cache and rescan circuit configs."""
        self._circuit_configs.clear()
        self._available_circuits = None
        self._scan_available_circuits()

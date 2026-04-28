#!/usr/bin/env python3
"""Stable CLI wrapper for the AnalogGym-Opt paper entry."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PAPER_CODE_RELATIVE = Path("Analoggym_opt_moo_Mahalanobis_paper") / "Analoggym_opt_moo_Mahalanobis_paper"
REQUIRED_FILE_PATH_KEYS = (
    "ACDC_cir_path",
    "Tran_cir_path",
    "netlist_path",
    "vars_path",
    "dev_params_path",
    "op_mean_std_path",
    "dc_results_path",
    "ac_results_path",
    "op_results_path",
    "GBW_PM_path",
    "tran_results_path",
    "tran_dat_path",
)
MODE_TO_QUICK_CONFIG = {
    "tt-only": {
        "enable_full_pvt_training": False,
        "enable_pvt_outer_loop": False,
    },
    "tt-proxy": {
        "enable_full_pvt_training": False,
        "enable_pvt_outer_loop": True,
    },
    "full-pvt": {
        "enable_full_pvt_training": True,
        "enable_pvt_outer_loop": False,
    },
}


def _json_dump(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _dedupe_paths(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    ordered: List[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return ordered


def resolve_repo_and_code_root(explicit_repo_root: str | None) -> Tuple[Path, Path]:
    candidates = []
    if explicit_repo_root:
        candidates.append(Path(explicit_repo_root).resolve())
    candidates.append(Path.cwd().resolve())
    candidates.extend(Path(__file__).resolve().parents)

    for repo_root in _dedupe_paths(candidates):
        code_root = repo_root / PAPER_CODE_RELATIVE
        if code_root.is_dir():
            return repo_root, code_root

    raise FileNotFoundError(
        "Could not find "
        f"{PAPER_CODE_RELATIVE}. Run from the repo root or pass --repo-root."
    )


def ensure_code_importable(code_root: Path) -> None:
    code_root_str = str(code_root)
    if code_root_str not in sys.path:
        sys.path.insert(0, code_root_str)


def import_from_code(code_root: Path, module_name: str):
    ensure_code_importable(code_root)
    return importlib.import_module(module_name)


def load_circuit_config(code_root: Path, circuit_name: str) -> Dict[str, Any]:
    loader_mod = import_from_code(code_root, "circuit_config_loader")
    loader = loader_mod.CircuitConfigLoader(config_dir=str(code_root / "circuit_configs"))
    return loader.get_circuit_config(circuit_name)


def list_circuits(code_root: Path) -> List[str]:
    loader_mod = import_from_code(code_root, "circuit_config_loader")
    loader = loader_mod.CircuitConfigLoader(config_dir=str(code_root / "circuit_configs"))
    return sorted(loader.get_available_circuits())


def count_design_variables(device_cfg: Dict[str, Any]) -> int:
    total = 0
    for item in (device_cfg or {}).values():
        total += len((item or {}).get("range", {}))
    return total


def build_circuit_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    graph = config.get("graph", {}) or {}
    performance = config.get("performance", {}) or {}
    paths = config.get("paths", {}) or {}
    device_cfg = config.get("device", {}) or {}

    return {
        "name": config.get("name"),
        "config_path": config.get("config_path"),
        "train_device": config.get("train_device"),
        "action_dim": config.get("action_dim"),
        "design_variable_count": count_design_variables(device_cfg),
        "device_groups": len(device_cfg),
        "device_names": list(device_cfg.keys()),
        "performance_metrics": list(performance.keys()),
        "post_target_bonus_metrics": list(((config.get("post_target_bonus") or {}).get("metrics") or [])),
        "graph": {
            "num_nodes": graph.get("num_nodes"),
            "num_relations": graph.get("num_relations"),
            "num_node_features": graph.get("num_node_features"),
            "edge_count": len(graph.get("edge_index", []) or []),
        },
        "simulation_root": str(Path(paths["ACDC_cir_path"]).resolve().parent) if paths.get("ACDC_cir_path") else None,
        "resolved_paths": {key: str(value) for key, value in paths.items()},
    }


def validate_circuit_config(config: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[str] = []
    warnings: List[str] = []

    for key in ("base_dir", "file_paths", "device", "performance", "action_dim", "ckt_hierarchy", "graph"):
        if key not in config:
            issues.append(f"Missing top-level key: {key}")

    paths = config.get("paths", {}) or {}
    if not paths:
        issues.append("Resolved paths are missing. The YAML could not be expanded correctly.")
    else:
        for key in REQUIRED_FILE_PATH_KEYS:
            path_value = paths.get(key)
            if not path_value:
                issues.append(f"Missing file_paths entry: {key}")
                continue
            if not Path(path_value).exists():
                issues.append(f"Path does not exist for {key}: {path_value}")

    device_cfg = config.get("device", {}) or {}
    expected_action_dim = count_design_variables(device_cfg)
    declared_action_dim = config.get("action_dim")
    if declared_action_dim != expected_action_dim:
        issues.append(
            "action_dim mismatch: "
            f"declared={declared_action_dim}, derived={expected_action_dim}"
        )

    graph = config.get("graph", {}) or {}
    for key in ("num_relations", "num_nodes", "num_node_features", "edge_index", "edge_type", "observation_matrix"):
        if key not in graph:
            issues.append(f"Missing graph key: {key}")

    edge_index = graph.get("edge_index", []) or []
    edge_type = graph.get("edge_type", []) or []
    if len(edge_index) != len(edge_type):
        issues.append(
            "Graph edge mismatch: "
            f"{len(edge_index)} edge_index entries vs {len(edge_type)} edge_type entries"
        )
    for idx, edge in enumerate(edge_index):
        if not isinstance(edge, list) or len(edge) != 2:
            issues.append(f"edge_index[{idx}] is not a 2-item list")
            break

    observation_matrix = graph.get("observation_matrix", {}) or {}
    feature_count = graph.get("num_node_features")
    if observation_matrix and graph.get("num_nodes") is not None and len(observation_matrix) != graph.get("num_nodes"):
        warnings.append(
            "observation_matrix size differs from num_nodes: "
            f"{len(observation_matrix)} vs {graph.get('num_nodes')}"
        )
    if observation_matrix and feature_count is not None:
        for name, values in observation_matrix.items():
            if len(values) != feature_count:
                issues.append(
                    f"Observation feature length mismatch for {name}: "
                    f"{len(values)} vs expected {feature_count}"
                )
                break

    performance = config.get("performance", {}) or {}
    if not performance:
        issues.append("performance block is empty")

    bonus_metrics = list(((config.get("post_target_bonus") or {}).get("metrics") or []))
    for metric in bonus_metrics:
        if metric not in performance:
            warnings.append(f"post_target_bonus metric not found in performance: {metric}")

    return {
        "circuit": config.get("name"),
        "valid": not issues,
        "issues": issues,
        "warnings": warnings,
    }


def print_circuit_summary(summary: Dict[str, Any]) -> None:
    print(f"Circuit: {summary['name']}")
    print(f"  Config: {summary['config_path']}")
    print(f"  Train device: {summary['train_device']}")
    print(f"  Action dim: {summary['action_dim']}")
    print(f"  Design variables: {summary['design_variable_count']}")
    print(f"  Device groups: {summary['device_groups']}")
    print(f"  Performance metrics: {', '.join(summary['performance_metrics'])}")
    bonus_metrics = summary.get("post_target_bonus_metrics") or []
    print(f"  Post-target bonus: {', '.join(bonus_metrics) if bonus_metrics else 'none'}")
    graph = summary.get("graph", {})
    print(
        "  Graph: "
        f"nodes={graph.get('num_nodes')}, "
        f"relations={graph.get('num_relations')}, "
        f"features={graph.get('num_node_features')}, "
        f"edges={graph.get('edge_count')}"
    )
    print(f"  Simulation root: {summary['simulation_root']}")


def print_validation_report(report: Dict[str, Any]) -> None:
    status = "VALID" if report["valid"] else "INVALID"
    print(f"Circuit {report['circuit']}: {status}")
    if report["issues"]:
        print("  Issues:")
        for issue in report["issues"]:
            print(f"    - {issue}")
    if report["warnings"]:
        print("  Warnings:")
        for warning in report["warnings"]:
            print(f"    - {warning}")
    if not report["issues"] and not report["warnings"]:
        print("  No structural issues found.")


def build_train_request(args: argparse.Namespace, repo_root: Path, code_root: Path) -> Dict[str, Any]:
    quick_config = {
        "num_steps": int(args.steps),
        **MODE_TO_QUICK_CONFIG[args.mode],
    }
    return {
        "repo_root": str(repo_root),
        "code_root": str(code_root),
        "circuit": args.circuit,
        "mode": args.mode,
        "quick_config": quick_config,
        "expected_training_root": str(code_root / "training_saves"),
    }


def snapshot_run_dirs(training_root: Path) -> Dict[str, float]:
    if not training_root.exists():
        return {}
    return {
        path.name: path.stat().st_mtime
        for path in training_root.iterdir()
        if path.is_dir()
    }


def detect_latest_run_dir(training_root: Path, before: Dict[str, float]) -> Path | None:
    if not training_root.exists():
        return None

    candidates = [path for path in training_root.iterdir() if path.is_dir()]
    if not candidates:
        return None

    new_dirs = [path for path in candidates if path.name not in before]
    target_pool = new_dirs if new_dirs else candidates
    return max(target_pool, key=lambda path: path.stat().st_mtime)


def print_run_artifacts(run_dir: Path) -> None:
    print(f"[wrapper] Run directory: {run_dir}")
    for subdir in (
        "top_designs_tt",
        "recommended_candidates_tt",
        "top_designs_verified_pvt",
        "recommended_candidates_verified_pvt",
    ):
        path = run_dir / subdir
        if path.exists():
            print(f"[wrapper] {subdir}: {path}")


@contextmanager
def pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def require_ngspice() -> None:
    if shutil.which("ngspice") is None:
        raise RuntimeError("ngspice was not found in PATH.")


def command_list_circuits(args: argparse.Namespace) -> int:
    _, code_root = resolve_repo_and_code_root(args.repo_root)
    circuits = list_circuits(code_root)
    if args.json:
        _json_dump({"circuits": circuits, "count": len(circuits), "code_root": str(code_root)})
        return 0

    print(f"Code root: {code_root}")
    for name in circuits:
        print(name)
    return 0


def command_describe_circuit(args: argparse.Namespace) -> int:
    _, code_root = resolve_repo_and_code_root(args.repo_root)
    config = load_circuit_config(code_root, args.circuit)
    summary = build_circuit_summary(config)
    summary["code_root"] = str(code_root)
    if args.json:
        _json_dump(summary)
        return 0

    print_circuit_summary(summary)
    return 0


def command_validate_circuit(args: argparse.Namespace) -> int:
    _, code_root = resolve_repo_and_code_root(args.repo_root)
    config = load_circuit_config(code_root, args.circuit)
    report = validate_circuit_config(config)
    report["code_root"] = str(code_root)
    if args.json:
        _json_dump(report)
    else:
        print_validation_report(report)
    return 0 if report["valid"] else 2


def command_train(args: argparse.Namespace) -> int:
    if args.json and not args.dry_run:
        raise SystemExit("--json is only supported with train when --dry-run is set.")

    repo_root, code_root = resolve_repo_and_code_root(args.repo_root)
    config = load_circuit_config(code_root, args.circuit)
    report = validate_circuit_config(config)
    if not args.skip_validate and not report["valid"]:
        print_validation_report(report)
        raise SystemExit("Circuit validation failed. Fix the YAML or rerun with --skip-validate.")

    request = build_train_request(args, repo_root, code_root)
    if args.dry_run:
        if args.json:
            _json_dump(request)
        else:
            print("Train dry-run:")
            print(f"  repo_root: {request['repo_root']}")
            print(f"  code_root: {request['code_root']}")
            print(f"  circuit: {request['circuit']}")
            print(f"  mode: {request['mode']}")
            print(f"  quick_config: {request['quick_config']}")
            print(f"  expected_training_root: {request['expected_training_root']}")
        return 0

    require_ngspice()
    main_mod = import_from_code(code_root, "main_AMP_grpo")
    training_root = code_root / "training_saves"
    before = snapshot_run_dirs(training_root)

    main_mod.CIRCUIT_NAME = args.circuit
    main_mod.QUICK_CONFIG = dict(request["quick_config"])

    with pushd(code_root):
        main_mod.main()

    run_dir = detect_latest_run_dir(training_root, before)
    if run_dir is not None:
        print_run_artifacts(run_dir)
    else:
        print("[wrapper] Training finished, but no run directory was detected.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI wrapper for the AnalogGym-Opt paper optimizer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--repo-root",
            help="Repo root. Needed if you run the skill outside this repository.",
        )
        subparser.add_argument(
            "--json",
            action="store_true",
            help="Print machine-readable JSON output when supported.",
        )

    list_parser = subparsers.add_parser(
        "list-circuits",
        help="List available circuit YAML configs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common(list_parser)
    list_parser.set_defaults(func=command_list_circuits)

    describe_parser = subparsers.add_parser(
        "describe-circuit",
        help="Show a compact summary for one circuit config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common(describe_parser)
    describe_parser.add_argument("--circuit", required=True, help="Circuit config name.")
    describe_parser.set_defaults(func=command_describe_circuit)

    validate_parser = subparsers.add_parser(
        "validate-circuit",
        help="Validate one circuit config before training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common(validate_parser)
    validate_parser.add_argument("--circuit", required=True, help="Circuit config name.")
    validate_parser.set_defaults(func=command_validate_circuit)

    train_parser = subparsers.add_parser(
        "train",
        help="Run the paper training entry with explicit circuit and mode.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common(train_parser)
    train_parser.add_argument("--circuit", required=True, help="Circuit config name.")
    train_parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Training steps forwarded to the paper quick config.",
    )
    train_parser.add_argument(
        "--mode",
        choices=sorted(MODE_TO_QUICK_CONFIG.keys()),
        default="tt-proxy",
        help="Training runtime mode.",
    )
    train_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved request without starting training.",
    )
    train_parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip circuit structure validation before training.",
    )
    train_parser.set_defaults(func=command_train)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

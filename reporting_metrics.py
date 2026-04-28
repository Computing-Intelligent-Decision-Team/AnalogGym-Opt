from typing import Any, Dict, List, Tuple


REPORTING_CONFIG_KEYS: Tuple[str, ...] = (
    "phase_margin",
    "dcgain",
    "PSRR",
    "cmrrdc",
    "settlingTime",
    "FOML",
    "FOMS",
    "Active_Area",
    "Power",
    "GBW",
    "sr",
)

REPORTING_RAW_ROWS: Tuple[Tuple[str, str], ...] = (
    ("phase_margin (deg)", "phase_margin"),
    ("dcgain", "dcgain"),
    ("PSRR", "PSRR"),
    ("cmrrdc", "cmrrdc"),
    ("setting_time", "settlingTime"),
    ("FOML", "FOML"),
    ("FOMS", "FOMS"),
    ("Active Area", "Active_Area"),
    ("Power", "Power"),
    ("GBW", "GBW"),
    ("sr", "sr"),
)

REPORTING_SCORE_KEYS: Tuple[str, ...] = (
    "phase_margin_score",
    "dcgain_score",
    "PSRR_score",
    "cmrrdc_score",
    "settlingTime_score",
    "FOML_score",
    "FOMS_score",
    "Active_Area_score",
    "Power_score",
    "GBW_score",
    "sr_score",
)

REPORTING_EXTRA_KEYS: Tuple[str, ...] = (
    "constraint_reward",
    "PM_violation",
)


def get_reporting_config_keys(performance_cfg: Dict[str, Any]) -> List[str]:
    return [key for key in REPORTING_CONFIG_KEYS if key in (performance_cfg or {})]


def get_reporting_metrics(perf: Dict[str, Any]) -> Dict[str, Any]:
    perf = perf or {}
    metrics: Dict[str, Any] = {}
    for perf_key, _ in REPORTING_RAW_ROWS:
        if perf_key in perf:
            metrics[perf_key] = perf[perf_key]
    return metrics


def get_reporting_signals(perf: Dict[str, Any]) -> Dict[str, Any]:
    perf = perf or {}
    signals: Dict[str, Any] = {}
    for key in REPORTING_EXTRA_KEYS:
        if key in perf:
            signals[key] = perf[key]
    return signals


def get_reporting_scores(perf: Dict[str, Any]) -> Dict[str, Any]:
    perf = perf or {}
    scores: Dict[str, Any] = {}
    for key in REPORTING_SCORE_KEYS:
        if key in perf:
            scores[key] = perf[key]
    return scores


def get_filtered_performance(perf: Dict[str, Any]) -> Dict[str, Any]:
    return get_reporting_metrics(perf)


def format_reporting_value(value: Any) -> str:
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if abs(numeric) >= 1e4 or (0 < abs(numeric) < 1e-3):
        return f"{numeric:.6e}"
    return f"{numeric:.6f}"


def iter_reporting_metric_items(perf: Dict[str, Any]) -> List[Tuple[str, Any]]:
    metrics = get_reporting_metrics(perf)
    items: List[Tuple[str, Any]] = []
    for perf_key, _ in REPORTING_RAW_ROWS:
        if perf_key in metrics:
            items.append((perf_key, metrics[perf_key]))
    return items


def iter_reporting_signal_items(perf: Dict[str, Any]) -> List[Tuple[str, Any]]:
    signals = get_reporting_signals(perf)
    return [(key, signals[key]) for key in REPORTING_EXTRA_KEYS if key in signals]


def build_reporting_table_rows(
    corner_info: Dict[str, Any],
    performance_cfg: Dict[str, Any],
) -> List[List[Any]]:
    corner_info = corner_info or {}
    performance_cfg = performance_cfg or {}
    rows: List[List[Any]] = []

    for perf_key, cfg_key in REPORTING_RAW_ROWS:
        if perf_key not in corner_info or cfg_key not in performance_cfg:
            continue
        rows.append(
            [
                perf_key,
                corner_info[perf_key],
                performance_cfg[cfg_key].get("target", ""),
            ]
        )

    return rows

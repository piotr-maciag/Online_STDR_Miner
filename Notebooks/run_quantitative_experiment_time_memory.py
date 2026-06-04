from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from math import log
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parent
if PROJECT_ROOT.name == "Notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))

import matplotlib.pyplot as plt
import pandas as pd
import psutil

sys.path.insert(0, str(PROJECT_ROOT / "Source"))

from spatio_temporal_network import (  # noqa: E402
    LearningParameters,
    OnlineSTDRMiner,
    initialize_network_from_events,
)


DATA_PATH = PROJECT_ROOT / "Data" / "Preprocessed" / "dataset_TSMC2014_NYC_preprocessed.csv"
OUTPUT_ROOT = PROJECT_ROOT / "Results" / "Quantitative_experiment_time_memory"

GRID_SHAPE = (10, 10)
ONLINE_SPATIAL_THRESHOLD_METERS = 10_000.0
ONLINE_ETA = 1.0
ONLINE_TAU_ACTIVATION = 60.0
ONLINE_TAU_REFRACTORY = 12
ONLINE_LAMBDA_DECAY = -log(0.10) / 120.0
DEFAULT_THETAS = (0.3,)
MAX_RULE_LENGTH = 2
MAX_SNAPSHOT_RULES = None

ONLINE_STDR_COLOR = "#4D4D4D"
MEMORY_COLOR = "#56B4E9"


def parse_optional_positive_int(value: str) -> int | None:
    if value.lower() in {"none", "all"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive, 'none', or 'all'")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Online STDR time/memory experiment with file-based progress logging.",
    )
    parser.add_argument(
        "--event-type-count",
        type=parse_optional_positive_int,
        default=None,
        help="Number of most frequent event types to use. Default: all.",
    )
    parser.add_argument(
        "--max-events",
        type=parse_optional_positive_int,
        default=None,
        help="Maximum number of event instances to process. Default: all.",
    )
    parser.add_argument(
        "--theta",
        type=float,
        action="append",
        default=None,
        help="Theta value to test. Can be passed multiple times. Default: 0.3.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Write one progress log line every N events. Default: 100.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1000,
        help="Write partial measurement CSV every N events. Default: 1000.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run directory name. Default: run_YYYYMMDD_HHMMSS.",
    )
    return parser.parse_args()


def configure_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("time_memory_experiment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(run_dir / "progress.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def load_selected_events(
    event_type_count: int | None,
    max_events: int | None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    raw_events = pd.read_csv(DATA_PATH).sort_values("timestamp", ignore_index=True)
    ranked_types = tuple(raw_events["Event_type"].value_counts().index)
    selected_types = ranked_types if event_type_count is None else ranked_types[:event_type_count]
    selected_events = raw_events[raw_events["Event_type"].isin(selected_types)].copy()
    selected_events = selected_events.sort_values("timestamp", ignore_index=True)
    if max_events is not None:
        selected_events = selected_events.head(max_events).copy()
    if selected_events.empty:
        raise ValueError("No events were selected. Check --event-type-count and --max-events.")

    selected_type_set = set(selected_events["Event_type"])
    selected_types_used = tuple(event_type for event_type in selected_types if event_type in selected_type_set)
    return selected_events.reset_index(drop=True), selected_types_used


def make_online_parameters(events: pd.DataFrame, online_theta: float) -> LearningParameters:
    observation_window_minutes = float(events["timestamp"].iloc[-1] - events["timestamp"].iloc[0])
    if observation_window_minutes <= 0:
        raise ValueError("The selected stream must cover a positive time span.")
    return LearningParameters(
        eta=ONLINE_ETA,
        lambda_decay=ONLINE_LAMBDA_DECAY,
        tau_activation=ONLINE_TAU_ACTIVATION,
        tau_refractory=ONLINE_TAU_REFRACTORY,
        theta=online_theta,
    )


def merged_event_type_rule_count(rules: list) -> int:
    return len({tuple(rule.event_type_rule) for rule in rules})


def rss_mb(process: psutil.Process) -> float:
    return process.memory_info().rss / (1024**2)


def relative_path(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def write_measurements(measurements: pd.DataFrame, output_dir: Path) -> None:
    measurements.to_csv(output_dir / "time_memory_measurements.csv", index=False)
    measurements.to_pickle(output_dir / "time_memory_measurements.pkl")


def run_online_time_memory(
    *,
    events: pd.DataFrame,
    selected_event_types: tuple[str, ...],
    online_theta: float,
    output_dir: Path,
    run_id: str,
    max_events_requested: int | None,
    config: dict,
    logger: logging.Logger,
    progress_every: int,
    save_every: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()

    setup_start = perf_counter()
    setup_memory_before_mb = rss_mb(process)
    parameters = make_online_parameters(events, online_theta)
    network = initialize_network_from_events(
        events,
        grid_shape=GRID_SHAPE,
        spatial_threshold=ONLINE_SPATIAL_THRESHOLD_METERS,
    )
    miner = OnlineSTDRMiner(network, parameters)
    setup_seconds = perf_counter() - setup_start
    setup_memory_after_mb = rss_mb(process)

    logger.info(
        "theta=%s network initialized: setup_seconds=%.6f rss_before=%.2fMB rss_after=%.2fMB %s",
        online_theta,
        setup_seconds,
        setup_memory_before_mb,
        setup_memory_after_mb,
        network.summary(),
    )

    rows: list[dict] = []
    cumulative_event_seconds = 0.0
    cumulative_process_seconds = 0.0
    cumulative_extract_seconds = 0.0
    peak_rss_mb = setup_memory_after_mb
    total_events = len(events)

    for event_index, event in enumerate(events.itertuples(index=False), start=1):
        event_series = pd.Series(event._asdict())

        event_start = perf_counter()
        process_start = perf_counter()
        miner.process_event(event_series, extract_rules=False)
        process_seconds = perf_counter() - process_start

        extract_start = perf_counter()
        rules = miner.extract_rules(
            max_rule_length=MAX_RULE_LENGTH,
            max_rules=MAX_SNAPSHOT_RULES,
        )
        extract_seconds = perf_counter() - extract_start
        event_seconds = perf_counter() - event_start

        current_rss_mb = rss_mb(process)
        peak_rss_mb = max(peak_rss_mb, current_rss_mb)
        cumulative_event_seconds += event_seconds
        cumulative_process_seconds += process_seconds
        cumulative_extract_seconds += extract_seconds

        rows.append(
            {
                "run_id": run_id,
                "algorithm": "online_stdr",
                "online_theta": online_theta,
                "event_index": event_index,
                "event_id": getattr(event, "Event_ID"),
                "event_type": getattr(event, "Event_type"),
                "event_timestamp": float(getattr(event, "timestamp")),
                "event_seconds": event_seconds,
                "process_event_seconds": process_seconds,
                "extract_rules_seconds": extract_seconds,
                "cumulative_event_seconds": cumulative_event_seconds,
                "cumulative_process_event_seconds": cumulative_process_seconds,
                "cumulative_extract_rules_seconds": cumulative_extract_seconds,
                "rss_mb": current_rss_mb,
                "rss_delta_from_setup_start_mb": current_rss_mb - setup_memory_before_mb,
                "peak_rss_mb": peak_rss_mb,
                "rule_count": len(rules),
                "merged_event_type_rule_count": merged_event_type_rule_count(rules),
                "significant_synapse_count": len(miner.significant_synapses),
                "stored_synapse_state_count": len(miner.network.synapse_states),
                "event_type_count": len(selected_event_types),
                "max_events_requested": max_events_requested,
            }
        )

        should_log = event_index % progress_every == 0 or event_index == total_events
        should_save = event_index % save_every == 0 or event_index == total_events

        if should_log:
            logger.info(
                "theta=%s events=%d/%d latest_event_seconds=%.6f cumulative_seconds=%.3f "
                "rss_mb=%.2f peak_rss_mb=%.2f rules=%d significant_synapses=%d",
                online_theta,
                event_index,
                total_events,
                event_seconds,
                cumulative_event_seconds,
                current_rss_mb,
                peak_rss_mb,
                len(rules),
                len(miner.significant_synapses),
            )
        if should_save:
            write_measurements(pd.DataFrame(rows), output_dir)

    measurements = pd.DataFrame(rows)
    final_rules = miner.extract_rules(max_rule_length=MAX_RULE_LENGTH, max_rules=MAX_SNAPSHOT_RULES)
    final_rules_frame = miner.rules_frame(final_rules)

    write_measurements(measurements, output_dir)
    final_rules_frame.to_csv(output_dir / "final_rules.csv", index=False)
    miner.significant_synapses_frame().to_csv(output_dir / "final_significant_synapses.csv", index=False)

    metadata = {
        **config,
        "online_theta": online_theta,
        "network": network.summary(),
        "setup_seconds": setup_seconds,
        "setup_memory_before_mb": setup_memory_before_mb,
        "setup_memory_after_mb": setup_memory_after_mb,
        "setup_memory_delta_mb": setup_memory_after_mb - setup_memory_before_mb,
        "final_cumulative_event_seconds": cumulative_event_seconds,
        "final_rss_mb": rss_mb(process),
        "final_peak_rss_mb": peak_rss_mb,
        "final_rule_count": len(final_rules),
        "final_merged_event_type_rule_count": merged_event_type_rule_count(final_rules),
        "final_significant_synapse_count": len(miner.significant_synapses),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("theta=%s completed", online_theta)
    return measurements, final_rules_frame


def save_plot(measurements_df: pd.DataFrame, run_dir: Path) -> Path:
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax_time = plt.subplots(figsize=(12, 6))
    ax_memory = ax_time.twinx()

    for online_theta, theta_rows in measurements_df.groupby("online_theta", sort=True):
        theta_rows = theta_rows.sort_values("event_index")
        ax_time.plot(
            theta_rows["event_index"],
            theta_rows["cumulative_event_seconds"],
            linewidth=1.6,
            color=ONLINE_STDR_COLOR,
            label=f"theta={online_theta:g} cumulative time",
        )
        ax_memory.plot(
            theta_rows["event_index"],
            theta_rows["rss_mb"],
            linewidth=1.2,
            linestyle="--",
            color=MEMORY_COLOR,
            label=f"theta={online_theta:g} RSS memory",
        )

    ax_time.set_title("Online STDR cumulative execution time and memory usage")
    ax_time.set_xlabel("Processed event instances")
    ax_time.set_ylabel("Cumulative execution time [s]")
    ax_memory.set_ylabel("RSS memory [MB]")
    ax_time.grid(True, alpha=0.25)

    handles_time, labels_time = ax_time.get_legend_handles_labels()
    handles_memory, labels_memory = ax_memory.get_legend_handles_labels()
    ax_time.legend(
        handles_time + handles_memory,
        labels_time + labels_memory,
        loc="upper left",
        frameon=False,
    )
    fig.tight_layout()

    plot_path = plot_dir / "cumulative_time_memory.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def save_summary(measurements_df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    summary_df = (
        measurements_df.sort_values(["online_theta", "event_index"])
        .groupby("online_theta", as_index=False)
        .tail(1)[
            [
                "online_theta",
                "event_index",
                "event_type_count",
                "cumulative_event_seconds",
                "cumulative_process_event_seconds",
                "cumulative_extract_rules_seconds",
                "rss_mb",
                "peak_rss_mb",
                "rule_count",
                "merged_event_type_rule_count",
                "significant_synapse_count",
                "stored_synapse_state_count",
            ]
        ]
        .sort_values("online_theta", ignore_index=True)
    )
    summary_df.to_csv(run_dir / "summary.csv", index=False)
    return summary_df


def main() -> None:
    args = parse_args()
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive.")
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive.")

    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "latest_run.txt").write_text(str(run_dir) + "\n", encoding="utf-8")

    logger = configure_logging(run_dir)
    logger.info("run_id=%s run_dir=%s", run_id, run_dir)

    selected_events, selected_types = load_selected_events(args.event_type_count, args.max_events)
    selection_dir = run_dir / "selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    selected_events.to_csv(selection_dir / "selected_events.csv", index=False)
    pd.DataFrame(
        {"rank": range(1, len(selected_types) + 1), "event_type": selected_types}
    ).to_csv(selection_dir / "selected_event_types.csv", index=False)

    online_thetas = tuple(args.theta or DEFAULT_THETAS)
    config = {
        "run_id": run_id,
        "data_path": relative_path(DATA_PATH),
        "event_type_count_requested": args.event_type_count,
        "event_type_count_used": len(selected_types),
        "max_events_requested": args.max_events,
        "event_count_used": len(selected_events),
        "online_thetas_to_run": list(online_thetas),
        "grid_shape": GRID_SHAPE,
        "online_spatial_threshold_meters": ONLINE_SPATIAL_THRESHOLD_METERS,
        "online_eta": ONLINE_ETA,
        "online_tau_activation": ONLINE_TAU_ACTIVATION,
        "online_tau_refractory": ONLINE_TAU_REFRACTORY,
        "online_lambda_decay": ONLINE_LAMBDA_DECAY,
        "max_rule_length": MAX_RULE_LENGTH,
        "max_snapshot_rules": MAX_SNAPSHOT_RULES,
        "progress_every": args.progress_every,
        "save_every": args.save_every,
    }
    (run_dir / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    logger.info("selected_events=%d selected_event_types=%d", len(selected_events), len(selected_types))

    all_measurements = []
    final_rules_by_theta = {}
    experiment_start = perf_counter()

    for online_theta in online_thetas:
        theta_label = f"theta_{online_theta:.2f}".replace(".", "_")
        measurements, final_rules = run_online_time_memory(
            events=selected_events,
            selected_event_types=selected_types,
            online_theta=online_theta,
            output_dir=run_dir / theta_label,
            run_id=run_id,
            max_events_requested=args.max_events,
            config=config,
            logger=logger,
            progress_every=args.progress_every,
            save_every=args.save_every,
        )
        all_measurements.append(measurements)
        final_rules_by_theta[online_theta] = final_rules

    measurements_df = pd.concat(all_measurements, ignore_index=True)
    write_measurements(measurements_df, run_dir)
    plot_path = save_plot(measurements_df, run_dir)
    save_summary(measurements_df, run_dir)

    elapsed_seconds = perf_counter() - experiment_start
    pd.DataFrame([{"run_id": run_id, "elapsed_seconds": elapsed_seconds}]).to_csv(
        run_dir / "run_metadata.csv",
        index=False,
    )
    logger.info("completed elapsed_seconds=%.3f plot=%s", elapsed_seconds, plot_path)


if __name__ == "__main__":
    main()

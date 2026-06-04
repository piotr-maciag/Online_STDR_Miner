from __future__ import annotations

from dataclasses import dataclass, field
import json
from math import asin, cos, exp, floor, radians, sin, sqrt
from pathlib import Path
from typing import Iterable

import pandas as pd


EARTH_RADIUS_METERS = 6_371_000.0


@dataclass(frozen=True)
class SpatialBounds:
    """Axis-aligned spatial extent covered by the event dataset."""

    min_longitude: float
    max_longitude: float
    min_latitude: float
    max_latitude: float

    @classmethod
    def from_events(
        cls,
        events: pd.DataFrame,
        longitude_col: str = "longitude",
        latitude_col: str = "latitude",
    ) -> "SpatialBounds":
        return cls(
            min_longitude=float(events[longitude_col].min()),
            max_longitude=float(events[longitude_col].max()),
            min_latitude=float(events[latitude_col].min()),
            max_latitude=float(events[latitude_col].max()),
        )


@dataclass(frozen=True)
class GridCell:
    cell_id: int
    row: int
    column: int
    center_longitude: float
    center_latitude: float

    @property
    def loc(self) -> tuple[float, float]:
        return (self.center_longitude, self.center_latitude)


@dataclass(frozen=True)
class Neuron:
    neuron_id: int
    cell_id: int
    event_type: str
    longitude: float
    latitude: float

    @property
    def loc(self) -> tuple[float, float]:
        return (self.longitude, self.latitude)


@dataclass
class SynapseState:
    weight: float = 0.0
    activation_trace: float = 0.0
    refractory_trace: float = 0.0
    last_update_time: float = 0.0


@dataclass(frozen=True)
class LearningParameters:
    eta: float
    lambda_decay: float
    tau_activation: float
    tau_refractory: float
    theta: float

    def __post_init__(self) -> None:
        if self.eta <= 0:
            raise ValueError("eta must be positive.")
        if self.lambda_decay <= 0:
            raise ValueError("lambda_decay must be positive.")
        if self.tau_activation <= 0:
            raise ValueError("tau_activation must be positive.")
        if self.tau_refractory <= 0:
            raise ValueError("tau_refractory must be positive.")
        if not 0 < self.theta < 1:
            raise ValueError("theta must be in the open interval (0, 1).")

    @property
    def max_weight(self) -> float:
        return self.eta / self.lambda_decay

    @property
    def significance_threshold(self) -> float:
        return self.theta * self.max_weight


@dataclass(frozen=True)
class DependencyRuleEndpoint:
    neuron_id: int
    event_type: str
    longitude: float
    latitude: float


@dataclass(frozen=True)
class SpatioTemporalDependencyRule:
    """One Spatio-Temporal Dependency Rule induced by a directed synapse."""

    endpoints: tuple[DependencyRuleEndpoint, ...]
    weights: tuple[float, ...]

    @property
    def significance(self) -> float:
        if not self.weights:
            return 0.0
        return min(self.weights)

    @property
    def event_type_rule(self) -> tuple[str, ...]:
        return tuple(endpoint.event_type for endpoint in self.endpoints)


@dataclass(frozen=True)
class DependencyRuleSnapshot:
    processed_events: int
    event_timestamp: float
    significant_synapse_count: int
    rules: tuple[SpatioTemporalDependencyRule, ...]


@dataclass
class SpatioTemporalNetwork:
    """Initialized network N from the manuscript's Section 3.2.

    The topology is explicit through incoming/outgoing adjacency lists. Synapse
    weights and traces are represented lazily: every connected synapse has zero
    state until a learning step stores a non-zero or recently updated value.
    """

    event_types: tuple[str, ...]
    grid_shape: tuple[int, int]
    bounds: SpatialBounds
    spatial_threshold: float
    cells: tuple[GridCell, ...]
    neurons: tuple[Neuron, ...]
    outgoing: dict[int, tuple[int, ...]]
    incoming: dict[int, tuple[int, ...]]
    neuron_index: dict[tuple[int, str], int]
    synapse_states: dict[tuple[int, int], SynapseState] = field(default_factory=dict)

    @property
    def neuron_count(self) -> int:
        return len(self.neurons)

    @property
    def synapse_count(self) -> int:
        return sum(len(postsynaptic) for postsynaptic in self.outgoing.values())

    @property
    def cell_count(self) -> int:
        return len(self.cells)

    def has_synapse(self, presynaptic_id: int, postsynaptic_id: int) -> bool:
        return postsynaptic_id in self.outgoing[presynaptic_id]

    def iter_synapses(self) -> Iterable[tuple[int, int]]:
        for presynaptic_id, postsynaptic_ids in self.outgoing.items():
            for postsynaptic_id in postsynaptic_ids:
                yield presynaptic_id, postsynaptic_id

    def incoming_synapses(self, neuron_id: int) -> list[tuple[int, int]]:
        return [(presynaptic_id, neuron_id) for presynaptic_id in self.incoming[neuron_id]]

    def outgoing_synapses(self, neuron_id: int) -> list[tuple[int, int]]:
        return [(neuron_id, postsynaptic_id) for postsynaptic_id in self.outgoing[neuron_id]]

    def get_synapse_state(self, presynaptic_id: int, postsynaptic_id: int) -> SynapseState:
        if not self.has_synapse(presynaptic_id, postsynaptic_id):
            raise KeyError(f"Synapse ({presynaptic_id}, {postsynaptic_id}) is not in N.")
        return self.synapse_states.get((presynaptic_id, postsynaptic_id), SynapseState())

    def set_synapse_state(
        self,
        presynaptic_id: int,
        postsynaptic_id: int,
        state: SynapseState,
    ) -> None:
        if not self.has_synapse(presynaptic_id, postsynaptic_id):
            raise KeyError(f"Synapse ({presynaptic_id}, {postsynaptic_id}) is not in N.")
        self.synapse_states[(presynaptic_id, postsynaptic_id)] = state

    def cell_id_for_location(self, longitude: float, latitude: float) -> int:
        rows, columns = self.grid_shape
        lon_span = self.bounds.max_longitude - self.bounds.min_longitude
        lat_span = self.bounds.max_latitude - self.bounds.min_latitude
        if lon_span <= 0 or lat_span <= 0:
            raise ValueError("Spatial bounds must have positive longitude and latitude spans.")

        column = min(
            columns - 1,
            max(0, floor((longitude - self.bounds.min_longitude) / lon_span * columns)),
        )
        row = min(
            rows - 1,
            max(0, floor((latitude - self.bounds.min_latitude) / lat_span * rows)),
        )
        return row * columns + column

    def neuron_id_for_event(self, event: pd.Series) -> int:
        cell_id = self.cell_id_for_location(float(event["longitude"]), float(event["latitude"]))
        return self.neuron_index[(cell_id, str(event["Event_type"]))]

    def summary(self) -> dict[str, int | float | tuple[int, int]]:
        return {
            "cells": self.cell_count,
            "event_types": len(self.event_types),
            "neurons": self.neuron_count,
            "synapses": self.synapse_count,
            "grid_shape": self.grid_shape,
            "spatial_threshold_meters": self.spatial_threshold,
        }


class OnlineSTDRMiner:
    """Online implementation of STDR discovery from the manuscript.

    Rule extraction returns significant Spatio-Temporal Dependency Rules
    (STDRs): one directed rule for each significant synapse.
    """

    def __init__(
        self,
        network: SpatioTemporalNetwork,
        parameters: LearningParameters,
    ) -> None:
        self.network = network
        self.parameters = parameters
        self.significant_synapses: set[tuple[int, int]] = set()
        self.last_spike_times: dict[int, float] = {}
        self.processed_events = 0

    def process_event(self, event: pd.Series, extract_rules: bool = False) -> list[SpatioTemporalDependencyRule]:
        spiking_neuron_id = self.network.neuron_id_for_event(event)
        current_time = float(event["timestamp"])
        updated_synapses = self.lazy_weight_update(spiking_neuron_id, current_time)

        threshold = self.parameters.significance_threshold
        for synapse in updated_synapses:
            state = self.network.get_synapse_state(*synapse)
            if state.weight >= threshold:
                self.significant_synapses.add(synapse)
            else:
                self.significant_synapses.discard(synapse)

        self.last_spike_times[spiking_neuron_id] = current_time
        self.processed_events += 1

        if extract_rules:
            return self.extract_rules()
        return []

    def process_events(
        self,
        events: pd.DataFrame,
        max_events: int | None = None,
        extract_rules_every: int | None = None,
    ) -> list[SpatioTemporalDependencyRule]:
        if max_events is not None:
            events = events.head(max_events)

        latest_rules: list[SpatioTemporalDependencyRule] = []
        for row_number, event in enumerate(events.sort_values("timestamp").itertuples(index=False), start=1):
            event_series = pd.Series(event._asdict())
            should_extract = (
                extract_rules_every is not None
                and row_number % extract_rules_every == 0
            )
            latest_rules = self.process_event(event_series, extract_rules=should_extract)

        if extract_rules_every is None:
            latest_rules = self.extract_rules()
        return latest_rules

    def rule_snapshots(
        self,
        events: pd.DataFrame,
        every: int = 1,
        max_events: int | None = None,
        max_rule_length: int | None = None,
        max_rules: int | None = None,
    ) -> Iterable[DependencyRuleSnapshot]:
        """Yield current dependency rules while processing the stream.

        This is the notebook-friendly form of Algorithm 1's Display(R) step.
        """
        if every <= 0:
            raise ValueError("every must be a positive integer.")
        max_rule_length = _normalize_optional_limit(max_rule_length, "max_rule_length")
        max_rules = _normalize_optional_limit(max_rules, "max_rules")
        if max_events is not None:
            events = events.head(max_events)

        for row_number, event in enumerate(events.sort_values("timestamp").itertuples(index=False), start=1):
            event_series = pd.Series(event._asdict())
            self.process_event(event_series, extract_rules=False)
            if row_number % every == 0:
                rules = self.extract_rules(
                    max_rule_length=max_rule_length,
                    max_rules=max_rules,
                )
                yield DependencyRuleSnapshot(
                    processed_events=self.processed_events,
                    event_timestamp=float(event_series["timestamp"]),
                    significant_synapse_count=len(self.significant_synapses),
                    rules=tuple(rules),
                )

    def lazy_weight_update(self, spiking_neuron_id: int, current_time: float) -> set[tuple[int, int]]:
        incoming = set(self.network.incoming_synapses(spiking_neuron_id))
        outgoing = set(self.network.outgoing_synapses(spiking_neuron_id))
        updated_synapses = incoming | outgoing | self.significant_synapses
        learning_synapses = {
            (presynaptic_id, postsynaptic_id)
            for presynaptic_id, postsynaptic_id in incoming
            if (
                self.last_spike_times.get(presynaptic_id) is not None
                and self.last_spike_times[presynaptic_id] < current_time
            )
        }

        delta_times = {}
        for synapse in updated_synapses:
            delta_times[synapse] = self._decay_synapse(
                synapse,
                current_time,
                decay_weight=synapse not in learning_synapses,
            )

        for presynaptic_id, postsynaptic_id in incoming:
            if (presynaptic_id, postsynaptic_id) in learning_synapses:
                state = self.network.get_synapse_state(presynaptic_id, postsynaptic_id)
                self._apply_weight_dynamics(
                    state=state,
                    delta_time=delta_times[(presynaptic_id, postsynaptic_id)],
                )
                state.refractory_trace = 1.0
                self.network.set_synapse_state(presynaptic_id, postsynaptic_id, state)

        for presynaptic_id, postsynaptic_id in outgoing:
            state = self.network.get_synapse_state(presynaptic_id, postsynaptic_id)
            state.activation_trace = 1.0
            self.network.set_synapse_state(presynaptic_id, postsynaptic_id, state)

        return updated_synapses

    def extract_rules(
        self,
        max_rule_length: int | None = None,
        max_rules: int | None = None,
    ) -> list[SpatioTemporalDependencyRule]:
        max_rule_length = _normalize_optional_limit(max_rule_length, "max_rule_length")
        max_rules = _normalize_optional_limit(max_rules, "max_rules")
        return extract_rules_from_synapses(
            network=self.network,
            synapses=self.significant_synapses,
            max_rule_length=max_rule_length,
            max_rules=max_rules,
        )

    def significant_synapses_frame(self) -> pd.DataFrame:
        rows = []
        for presynaptic_id, postsynaptic_id in sorted(self.significant_synapses):
            presynaptic = self.network.neurons[presynaptic_id]
            postsynaptic = self.network.neurons[postsynaptic_id]
            state = self.network.get_synapse_state(presynaptic_id, postsynaptic_id)
            rows.append(
                {
                    "presynaptic_id": presynaptic_id,
                    "postsynaptic_id": postsynaptic_id,
                    "presynaptic_event_type": presynaptic.event_type,
                    "postsynaptic_event_type": postsynaptic.event_type,
                    "presynaptic_cell_id": presynaptic.cell_id,
                    "postsynaptic_cell_id": postsynaptic.cell_id,
                    "weight": state.weight,
                    "weight_ratio": state.weight / self.parameters.max_weight,
                }
            )
        if not rows:
            return pd.DataFrame(
                columns=[
                    "presynaptic_id",
                    "postsynaptic_id",
                    "presynaptic_event_type",
                    "postsynaptic_event_type",
                    "presynaptic_cell_id",
                    "postsynaptic_cell_id",
                    "weight",
                    "weight_ratio",
                ]
            )
        return pd.DataFrame(rows).sort_values("weight", ascending=False, ignore_index=True)

    def rules_frame(self, rules: Iterable[SpatioTemporalDependencyRule]) -> pd.DataFrame:
        rows = []
        for rule in rules:
            rows.append(
                {
                    "length": len(rule.endpoints),
                    "significance": rule.significance,
                    "significance_ratio": rule.significance / self.parameters.max_weight,
                    "event_type_rule": " -> ".join(rule.event_type_rule),
                    "presynaptic_id": rule.endpoints[0].neuron_id,
                    "postsynaptic_id": rule.endpoints[-1].neuron_id,
                    "presynaptic_event_type": rule.endpoints[0].event_type,
                    "postsynaptic_event_type": rule.endpoints[-1].event_type,
                    "presynaptic_location": (
                        rule.endpoints[0].longitude,
                        rule.endpoints[0].latitude,
                    ),
                    "postsynaptic_location": (
                        rule.endpoints[-1].longitude,
                        rule.endpoints[-1].latitude,
                    ),
                    "locations": [
                        (endpoint.longitude, endpoint.latitude)
                        for endpoint in rule.endpoints
                    ],
                    "weights": rule.weights,
                    "weight_ratios": tuple(
                        weight / self.parameters.max_weight
                        for weight in rule.weights
                    ),
                }
            )
        if not rows:
            return pd.DataFrame(
                columns=[
                    "length",
                    "significance",
                    "significance_ratio",
                    "event_type_rule",
                    "presynaptic_id",
                    "postsynaptic_id",
                    "presynaptic_event_type",
                    "postsynaptic_event_type",
                    "presynaptic_location",
                    "postsynaptic_location",
                    "locations",
                    "weights",
                    "weight_ratios",
                ]
            )
        return pd.DataFrame(rows).sort_values(
            ["significance", "length"],
            ascending=[False, False],
            ignore_index=True,
        )

    def dump_rule_snapshot(
        self,
        snapshot: DependencyRuleSnapshot,
        output_dir: Path | str,
        prefix: str = "snapshot",
    ) -> tuple[Path, Path]:
        """Write one online dependency-rule snapshot to JSON metadata and CSV rules."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = f"{prefix}_{snapshot.processed_events:06d}"
        metadata_path = output_dir / f"{stem}_metadata.json"
        rules_path = output_dir / f"{stem}_rules.csv"

        metadata = {
            "processed_events": snapshot.processed_events,
            "event_timestamp": snapshot.event_timestamp,
            "significant_synapse_count": snapshot.significant_synapse_count,
            "rule_count": len(snapshot.rules),
            "network": self.network.summary(),
            "learning": {
                "eta": self.parameters.eta,
                "lambda_decay": self.parameters.lambda_decay,
                "tau_activation": self.parameters.tau_activation,
                "tau_refractory": self.parameters.tau_refractory,
                "theta": self.parameters.theta,
                "max_weight": self.parameters.max_weight,
                "significance_threshold": self.parameters.significance_threshold,
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.rules_frame(snapshot.rules).to_csv(rules_path, index=False)
        return metadata_path, rules_path

    def _apply_weight_dynamics(self, state: SynapseState, delta_time: float) -> None:
        if delta_time <= 0:
            return

        rate = self.parameters.lambda_decay + self.parameters.eta * state.refractory_trace
        target_weight = self.parameters.eta * state.activation_trace / rate
        relaxation = 1.0 - exp(-rate * delta_time)
        state.weight += (target_weight - state.weight) * relaxation
        state.weight = max(0.0, state.weight)

    def _decay_synapse(
        self,
        synapse: tuple[int, int],
        current_time: float,
        decay_weight: bool = True,
    ) -> float:
        state = self.network.get_synapse_state(*synapse)
        delta_time = current_time - state.last_update_time
        if delta_time < 0:
            raise ValueError("Events must be processed in non-decreasing timestamp order.")

        state.activation_trace *= exp(-delta_time / self.parameters.tau_activation)
        state.refractory_trace *= exp(-delta_time / self.parameters.tau_refractory)
        if decay_weight:
            state.weight *= exp(-self.parameters.lambda_decay * delta_time)
        state.last_update_time = current_time
        self.network.set_synapse_state(*synapse, state)
        return delta_time


def clear_rule_snapshot_dumps(output_dir: Path | str, prefix: str = "snapshot") -> None:
    """Remove previous snapshot dump files created by dump_rule_snapshot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob(f"{prefix}_*_metadata.json"):
        path.unlink()
    for path in output_dir.glob(f"{prefix}_*_rules.csv"):
        path.unlink()


def extract_rules_from_synapses(
    network: SpatioTemporalNetwork,
    synapses: Iterable[tuple[int, int]],
    max_rule_length: int | None = None,
    max_rules: int | None = None,
) -> list[SpatioTemporalDependencyRule]:
    """Extract significant STDRs induced by significant synapses.

    Definition 3.2 defines an STDR as one directed dependency
    ``(event type, location) -> (event type, location)`` whose significance is
    the synaptic weight. Therefore each significant synapse induces exactly one
    output rule; longer graph paths are not expanded in the STDR model.
    """
    max_rule_length = _normalize_optional_limit(max_rule_length, "max_rule_length")
    max_rules = _normalize_optional_limit(max_rules, "max_rules")
    if max_rule_length is not None and max_rule_length < 2:
        return []

    rules = [
        _rule_from_synapse(network, presynaptic_id, postsynaptic_id)
        for presynaptic_id, postsynaptic_id in set(synapses)
    ]
    rules.sort(
        key=lambda rule: (
            -rule.significance,
            rule.event_type_rule,
            tuple(endpoint.neuron_id for endpoint in rule.endpoints),
        )
    )

    if max_rules is not None:
        return rules[:max_rules]
    return rules


def _normalize_optional_limit(value: int | None, name: str) -> int | None:
    if value is None or value == -1:
        return None
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, -1, or None.")
    return value


def _rule_from_synapse(
    network: SpatioTemporalNetwork,
    presynaptic_id: int,
    postsynaptic_id: int,
) -> SpatioTemporalDependencyRule:
    return _rule_from_path(network, (presynaptic_id, postsynaptic_id))


def _rule_from_path(
    network: SpatioTemporalNetwork,
    path: tuple[int, ...],
) -> SpatioTemporalDependencyRule:
    endpoints = tuple(
        DependencyRuleEndpoint(
            neuron_id=neuron.neuron_id,
            event_type=neuron.event_type,
            longitude=neuron.longitude,
            latitude=neuron.latitude,
        )
        for neuron in (network.neurons[neuron_id] for neuron_id in path)
    )
    weights = tuple(
        network.get_synapse_state(path[index], path[index + 1]).weight
        for index in range(len(path) - 1)
    )
    return SpatioTemporalDependencyRule(endpoints=endpoints, weights=weights)


def initialize_network_from_csv(
    csv_path: Path | str,
    grid_shape: tuple[int, int],
    spatial_threshold: float,
    longitude_col: str = "longitude",
    latitude_col: str = "latitude",
    event_type_col: str = "Event_type",
    bounds: SpatialBounds | None = None,
) -> SpatioTemporalNetwork:
    """Load a preprocessed event CSV and initialize network N.

    ``spatial_threshold`` is interpreted in meters for longitude/latitude data.
    """
    events = pd.read_csv(csv_path, usecols=[longitude_col, latitude_col, event_type_col])
    return initialize_network_from_events(
        events=events,
        grid_shape=grid_shape,
        spatial_threshold=spatial_threshold,
        longitude_col=longitude_col,
        latitude_col=latitude_col,
        event_type_col=event_type_col,
        bounds=bounds,
    )


def initialize_network_from_events(
    events: pd.DataFrame,
    grid_shape: tuple[int, int],
    spatial_threshold: float,
    longitude_col: str = "longitude",
    latitude_col: str = "latitude",
    event_type_col: str = "Event_type",
    bounds: SpatialBounds | None = None,
) -> SpatioTemporalNetwork:
    """Initialize neurons and bidirectional synapses according to the paper.

    For each grid cell, one neuron is created for each event type. Inside a cell,
    neurons are fully connected in both directions, excluding self-synapses.
    Between two cells whose centers are no farther than ``spatial_threshold``
    meters, neurons are connected in both directions when they represent
    different event types. Synapses between neurons of the same event type are
    not created.
    """
    _validate_grid_shape(grid_shape)
    if spatial_threshold < 0:
        raise ValueError("spatial_threshold must be non-negative.")

    required_columns = {longitude_col, latitude_col, event_type_col}
    missing_columns = required_columns.difference(events.columns)
    if missing_columns:
        raise ValueError(f"Missing required event columns: {sorted(missing_columns)}")

    bounds = bounds or SpatialBounds.from_events(events, longitude_col, latitude_col)
    cells = _build_grid_cells(grid_shape, bounds)
    event_types = tuple(sorted(str(value) for value in events[event_type_col].dropna().unique()))
    if not event_types:
        raise ValueError("At least one event type is required to initialize N.")

    neurons, neuron_index, cell_neuron_ids = _build_neurons(cells, event_types)
    outgoing_sets = {neuron.neuron_id: set() for neuron in neurons}
    incoming_sets = {neuron.neuron_id: set() for neuron in neurons}

    for cell in cells:
        _connect_neuron_groups(
            cell_neuron_ids[cell.cell_id],
            cell_neuron_ids[cell.cell_id],
            neurons,
            outgoing_sets,
            incoming_sets,
            include_self=False,
        )

    for left_index, left_cell in enumerate(cells):
        for right_cell in cells[left_index + 1 :]:
            distance = haversine_distance_meters(
                left_cell.center_longitude,
                left_cell.center_latitude,
                right_cell.center_longitude,
                right_cell.center_latitude,
            )
            if distance <= spatial_threshold:
                _connect_neuron_groups(
                    cell_neuron_ids[left_cell.cell_id],
                    cell_neuron_ids[right_cell.cell_id],
                    neurons,
                    outgoing_sets,
                    incoming_sets,
                    include_self=True,
                )
                _connect_neuron_groups(
                    cell_neuron_ids[right_cell.cell_id],
                    cell_neuron_ids[left_cell.cell_id],
                    neurons,
                    outgoing_sets,
                    incoming_sets,
                    include_self=True,
                )

    outgoing = {
        neuron_id: tuple(sorted(postsynaptic_ids))
        for neuron_id, postsynaptic_ids in outgoing_sets.items()
    }
    incoming = {
        neuron_id: tuple(sorted(presynaptic_ids))
        for neuron_id, presynaptic_ids in incoming_sets.items()
    }

    return SpatioTemporalNetwork(
        event_types=event_types,
        grid_shape=grid_shape,
        bounds=bounds,
        spatial_threshold=spatial_threshold,
        cells=cells,
        neurons=neurons,
        outgoing=outgoing,
        incoming=incoming,
        neuron_index=neuron_index,
    )


def _validate_grid_shape(grid_shape: tuple[int, int]) -> None:
    if len(grid_shape) != 2:
        raise ValueError("grid_shape must be a pair: (rows, columns).")
    rows, columns = grid_shape
    if rows <= 0 or columns <= 0:
        raise ValueError("grid_shape rows and columns must be positive integers.")


def haversine_distance_meters(
    longitude_a: float,
    latitude_a: float,
    longitude_b: float,
    latitude_b: float,
) -> float:
    """Return great-circle distance between two lon/lat points in meters."""
    lon_a = radians(longitude_a)
    lat_a = radians(latitude_a)
    lon_b = radians(longitude_b)
    lat_b = radians(latitude_b)

    delta_lon = lon_b - lon_a
    delta_lat = lat_b - lat_a
    haversine = (
        sin(delta_lat / 2.0) ** 2
        + cos(lat_a) * cos(lat_b) * sin(delta_lon / 2.0) ** 2
    )
    central_angle = 2.0 * asin(sqrt(haversine))
    return EARTH_RADIUS_METERS * central_angle


def _build_grid_cells(grid_shape: tuple[int, int], bounds: SpatialBounds) -> tuple[GridCell, ...]:
    rows, columns = grid_shape
    lon_span = bounds.max_longitude - bounds.min_longitude
    lat_span = bounds.max_latitude - bounds.min_latitude
    if lon_span <= 0 or lat_span <= 0:
        raise ValueError("Spatial bounds must have positive longitude and latitude spans.")

    cell_width = lon_span / columns
    cell_height = lat_span / rows
    cells = []
    for row in range(rows):
        for column in range(columns):
            cell_id = row * columns + column
            cells.append(
                GridCell(
                    cell_id=cell_id,
                    row=row,
                    column=column,
                    center_longitude=bounds.min_longitude + (column + 0.5) * cell_width,
                    center_latitude=bounds.min_latitude + (row + 0.5) * cell_height,
                )
            )
    return tuple(cells)


def _build_neurons(
    cells: tuple[GridCell, ...],
    event_types: tuple[str, ...],
) -> tuple[tuple[Neuron, ...], dict[tuple[int, str], int], dict[int, list[int]]]:
    neurons = []
    neuron_index = {}
    cell_neuron_ids = {cell.cell_id: [] for cell in cells}
    for cell in cells:
        for event_type in event_types:
            neuron_id = len(neurons)
            neurons.append(
                Neuron(
                    neuron_id=neuron_id,
                    cell_id=cell.cell_id,
                    event_type=event_type,
                    longitude=cell.center_longitude,
                    latitude=cell.center_latitude,
                )
            )
            neuron_index[(cell.cell_id, event_type)] = neuron_id
            cell_neuron_ids[cell.cell_id].append(neuron_id)
    return tuple(neurons), neuron_index, cell_neuron_ids


def _connect_neuron_groups(
    presynaptic_ids: list[int],
    postsynaptic_ids: list[int],
    neurons: tuple[Neuron, ...],
    outgoing_sets: dict[int, set[int]],
    incoming_sets: dict[int, set[int]],
    include_self: bool,
) -> None:
    for presynaptic_id in presynaptic_ids:
        for postsynaptic_id in postsynaptic_ids:
            if (
                (include_self or presynaptic_id != postsynaptic_id)
                and neurons[presynaptic_id].event_type != neurons[postsynaptic_id].event_type
            ):
                outgoing_sets[presynaptic_id].add(postsynaptic_id)
                incoming_sets[postsynaptic_id].add(presynaptic_id)

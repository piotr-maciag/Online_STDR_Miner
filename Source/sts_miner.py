from __future__ import annotations

from dataclasses import dataclass
from math import asin, cos, pi, radians, sin, sqrt
from typing import Iterable

import pandas as pd


EARTH_RADIUS_METERS = 6_371_000.0


@dataclass(frozen=True)
class EventInstance:
    """One spatio-temporal event instance used by STS-Miner."""

    event_id: int
    event_type: str
    longitude: float
    latitude: float
    x: float
    y: float
    time: float


@dataclass(frozen=True)
class FollowJoinResult:
    """Output of a follow join between a tail event set and one event type."""

    join_size: int
    tail_events: tuple[EventInstance, ...]


@dataclass(frozen=True)
class SequencePattern:
    """A discovered STS-Miner pattern."""

    sequence: tuple[str, ...]
    sequence_index: float
    tail_event_count: int
    last_density_ratio: float


def add_local_meter_coordinates(
    events: pd.DataFrame,
    *,
    longitude_col: str = "longitude",
    latitude_col: str = "latitude",
    x_col: str = "x_meters",
    y_col: str = "y_meters",
) -> pd.DataFrame:
    """Return a copy of events with local equirectangular meter coordinates.

    The miner uses longitude/latitude with haversine distance for neighborhood
    checks. These planar coordinates are retained only for estimating the
    embedding volume used by the density ratio. This helper prepares a
    DataFrame supplied by the caller; it does not read any dataset from disk.
    """

    required_columns = {longitude_col, latitude_col}
    missing = required_columns.difference(events.columns)
    if missing:
        raise ValueError(f"Missing required coordinate columns: {sorted(missing)}")

    projected = events.copy()
    origin_lon = float(projected[longitude_col].min())
    origin_lat = float(projected[latitude_col].min())
    mean_lat_rad = radians(float(projected[latitude_col].mean()))

    projected[x_col] = (
        (projected[longitude_col].astype(float) - origin_lon)
        .map(radians)
        .mul(EARTH_RADIUS_METERS * cos(mean_lat_rad))
    )
    projected[y_col] = (
        (projected[latitude_col].astype(float) - origin_lat)
        .map(radians)
        .mul(EARTH_RADIUS_METERS)
    )
    return projected


class STSMiner:
    """Basic STS-Miner using plane-sweep follow joins.

    This is the non-microclustering path from the archived implementation.
    It groups event instances by type, expands sequences depth-first, computes
    follow neighborhoods with a temporal plane sweep and haversine spatial
    distance, and scores sequences by the minimum density ratio along the
    sequence.
    """

    def __init__(
        self,
        events: pd.DataFrame,
        *,
        spatial_radius: float,
        temporal_window: float,
        min_sequence_index: float = 1.0,
        max_length: int | None = 4,
        event_id_col: str = "Event_ID",
        event_type_col: str = "Event_type",
        longitude_col: str = "longitude",
        latitude_col: str = "latitude",
        x_col: str = "x_meters",
        y_col: str = "y_meters",
        time_col: str = "timestamp",
    ) -> None:
        if spatial_radius <= 0:
            raise ValueError("spatial_radius must be positive.")
        if temporal_window <= 0:
            raise ValueError("temporal_window must be positive.")
        if min_sequence_index < 0:
            raise ValueError("min_sequence_index must be non-negative.")
        if max_length is not None and max_length < 2:
            raise ValueError("max_length must be at least 2.")
        self.spatial_radius = float(spatial_radius)
        self.temporal_window = float(temporal_window)
        self.min_sequence_index = float(min_sequence_index)
        self.max_length = None if max_length is None else int(max_length)

        self.events_by_type = self._build_event_index(
            events,
            event_id_col=event_id_col,
            event_type_col=event_type_col,
            longitude_col=longitude_col,
            latitude_col=latitude_col,
            x_col=x_col,
            y_col=y_col,
            time_col=time_col,
        )
        self.event_types = tuple(self.events_by_type)
        self.embedding_volume = self._embedding_volume()
        self.neighborhood_volume = pi * self.spatial_radius * self.spatial_radius * self.temporal_window
        self.patterns: list[SequencePattern] = []

    def mine(self) -> list[SequencePattern]:
        """Run STS-Miner and return patterns sorted by decreasing sequence index."""

        self.patterns = []
        for event_type, event_set in self.events_by_type.items():
            self._expand(
                sequence=(event_type,),
                tail_events=event_set,
                sequence_index=float("inf"),
            )

        self.patterns.sort(key=lambda pattern: (-pattern.sequence_index, pattern.sequence))
        return self.patterns

    def _expand(
        self,
        *,
        sequence: tuple[str, ...],
        tail_events: tuple[EventInstance, ...],
        sequence_index: float,
    ) -> None:
        if self.max_length is not None and len(sequence) >= self.max_length:
            return
        if not tail_events:
            return

        for next_type in self.event_types:
            if next_type == sequence[-1]:
                continue

            next_event_set = self.events_by_type[next_type]
            join_result = plane_sweep_follow_join(
                tail_events,
                next_event_set,
                spatial_radius=self.spatial_radius,
                temporal_window=self.temporal_window,
            )
            density_ratio = self._density_ratio(
                tail_count=len(tail_events),
                join_size=join_result.join_size,
                event_type_count=len(next_event_set),
            )
            new_sequence_index = min(sequence_index, density_ratio)

            if new_sequence_index >= self.min_sequence_index and join_result.tail_events:
                new_sequence = sequence + (next_type,)
                self.patterns.append(
                    SequencePattern(
                        sequence=new_sequence,
                        sequence_index=new_sequence_index,
                        tail_event_count=len(join_result.tail_events),
                        last_density_ratio=density_ratio,
                    )
                )
                self._expand(
                    sequence=new_sequence,
                    tail_events=join_result.tail_events,
                    sequence_index=new_sequence_index,
                )

    def _density_ratio(self, *, tail_count: int, join_size: int, event_type_count: int) -> float:
        if tail_count == 0 or event_type_count == 0 or self.embedding_volume == 0:
            return 0.0

        average_neighborhood_density = join_size / self.neighborhood_volume / tail_count
        global_density = event_type_count / self.embedding_volume
        if global_density == 0:
            return 0.0
        return average_neighborhood_density / global_density

    def _embedding_volume(self) -> float:
        all_events = [event for events in self.events_by_type.values() for event in events]
        if not all_events:
            return 0.0

        x_span = max(event.x for event in all_events) - min(event.x for event in all_events)
        y_span = max(event.y for event in all_events) - min(event.y for event in all_events)
        time_span = max(event.time for event in all_events) - min(event.time for event in all_events)
        return max(x_span, 1.0) * max(y_span, 1.0) * max(time_span, 1.0)

    @staticmethod
    def _build_event_index(
        events: pd.DataFrame,
        *,
        event_id_col: str,
        event_type_col: str,
        longitude_col: str,
        latitude_col: str,
        x_col: str,
        y_col: str,
        time_col: str,
    ) -> dict[str, tuple[EventInstance, ...]]:
        required_columns = {
            event_id_col,
            event_type_col,
            longitude_col,
            latitude_col,
            x_col,
            y_col,
            time_col,
        }
        missing = required_columns.difference(events.columns)
        if missing:
            raise ValueError(f"Missing required event columns: {sorted(missing)}")

        indexed: dict[str, list[EventInstance]] = {}
        for _, row in events.iterrows():
            event = EventInstance(
                event_id=int(row[event_id_col]),
                event_type=str(row[event_type_col]),
                longitude=float(row[longitude_col]),
                latitude=float(row[latitude_col]),
                x=float(row[x_col]),
                y=float(row[y_col]),
                time=float(row[time_col]),
            )
            indexed.setdefault(event.event_type, []).append(event)

        return {
            event_type: tuple(sorted(instances, key=lambda event: event.time))
            for event_type, instances in sorted(indexed.items())
        }


def plane_sweep_follow_join(
    tail_events: Iterable[EventInstance],
    candidate_events: Iterable[EventInstance],
    *,
    spatial_radius: float,
    temporal_window: float,
) -> FollowJoinResult:
    """Find candidate events that follow a tail event set.

    The sweep line advances through candidate events by time. For each tail
    event p, only candidates q with p.time < q.time <= p.time + temporal_window
    are spatially checked with exact haversine distance.
    """

    tails = sorted(tail_events, key=lambda event: event.time)
    candidates = tuple(sorted(candidate_events, key=lambda event: event.time))

    join_size = 0
    tail_by_id: dict[int, EventInstance] = {}
    active_start = 0
    active_end = 0

    for tail in tails:
        while active_start < len(candidates) and candidates[active_start].time <= tail.time:
            active_start += 1
        if active_end < active_start:
            active_end = active_start
        while active_end < len(candidates) and candidates[active_end].time <= tail.time + temporal_window:
            active_end += 1

        for candidate_index in range(active_start, active_end):
            candidate = candidates[candidate_index]
            distance = haversine_distance_meters(
                tail.longitude,
                tail.latitude,
                candidate.longitude,
                candidate.latitude,
            )
            if distance <= spatial_radius:
                join_size += 1
                tail_by_id[candidate.event_id] = candidate

    tail_events_result = tuple(sorted(tail_by_id.values(), key=lambda event: event.time))
    return FollowJoinResult(join_size=join_size, tail_events=tail_events_result)


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


def patterns_to_frame(patterns: Iterable[SequencePattern]) -> pd.DataFrame:
    """Convert discovered patterns to an analysis-friendly DataFrame."""

    return pd.DataFrame(
        [
            {
                "sequence": " -> ".join(pattern.sequence),
                "length": len(pattern.sequence),
                "sequence_index": pattern.sequence_index,
                "tail_event_count": pattern.tail_event_count,
                "last_density_ratio": pattern.last_density_ratio,
            }
            for pattern in patterns
        ]
    )

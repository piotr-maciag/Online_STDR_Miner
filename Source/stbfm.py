from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from sts_miner import EventInstance, add_local_meter_coordinates, plane_sweep_follow_join


@dataclass(frozen=True)
class STBFMPattern:
    """A PI-strong spatio-temporal sequential pattern discovered by STBFM."""

    sequence: tuple[str, ...]
    participation_index: float
    tail_event_count: int
    last_participation_ratio: float

    @property
    def length(self) -> int:
        return len(self.sequence)


@dataclass
class _PatternNode:
    event_type: str
    tail_events: tuple[EventInstance, ...]
    participation_index: float
    first_parent: "_PatternNode | None" = None
    second_parent: "_PatternNode | None" = None
    children: list["_PatternNode"] = field(default_factory=list)

    @property
    def sequence(self) -> tuple[str, ...]:
        nodes: list[_PatternNode] = []
        current: _PatternNode | None = self
        while current is not None:
            nodes.append(current)
            current = current.first_parent
        return tuple(node.event_type for node in reversed(nodes))

    @property
    def length(self) -> int:
        return len(self.sequence)


class STBFM:
    """Breadth-first STBFM miner for all PI-strong ST-sequential patterns.

    This implements the non-closed STBFM/CSP-tree strategy from the archived
    C++ code and manuscript. A node stores only the last event type, its tail
    event set, both immediate parents, and children. Candidate generation
    extends each level-k pattern with children of its second parent, while
    candidate verification computes the next tail event set with the same
    temporal plane-sweep follow join style used by ``STSMiner``.
    """

    def __init__(
        self,
        events: pd.DataFrame,
        *,
        spatial_radius: float,
        temporal_window: float,
        min_participation_index: float,
        max_length: int | None = 4,
        include_singletons: bool = False,
        threshold_inclusive: bool = False,
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
        if not 0 <= min_participation_index <= 1:
            raise ValueError("min_participation_index must be in the interval [0, 1].")
        if max_length is not None and max_length < 2:
            raise ValueError("max_length must be at least 2.")

        self.spatial_radius = float(spatial_radius)
        self.temporal_window = float(temporal_window)
        self.min_participation_index = float(min_participation_index)
        self.max_length = None if max_length is None else int(max_length)
        self.include_singletons = include_singletons
        self.threshold_inclusive = threshold_inclusive

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
        self.event_type_counts = {
            event_type: len(event_set)
            for event_type, event_set in self.events_by_type.items()
        }
        self.levels: list[list[_PatternNode]] = []
        self.patterns: list[STBFMPattern] = []

    def mine(self) -> list[STBFMPattern]:
        """Run STBFM and return PI-strong patterns sorted by PI and sequence."""

        self.patterns = []
        self.levels = [self._make_singleton_level()]

        if self.include_singletons:
            self.patterns.extend(
                self._to_pattern(node, last_participation_ratio=1.0)
                for node in self.levels[0]
                if self._passes_threshold(node.participation_index)
            )

        if self.max_length is None or self.max_length >= 2:
            level = self._make_length_two_level()
            if level:
                self.levels.append(level)
                self.patterns.extend(
                    self._to_pattern(node, last_participation_ratio=node.participation_index)
                    for node in level
                )

        while self.levels[-1] and (
            self.max_length is None or self.levels[-1][0].length < self.max_length
        ):
            next_level = self._generate_next_level(self.levels[-1])
            if not next_level:
                break
            self.levels.append(next_level)
            self.patterns.extend(self._to_pattern(node) for node in next_level)

        self.patterns.sort(
            key=lambda pattern: (
                -pattern.participation_index,
                pattern.length,
                pattern.sequence,
            )
        )
        return self.patterns

    def _make_singleton_level(self) -> list[_PatternNode]:
        return [
            _PatternNode(
                event_type=event_type,
                tail_events=event_set,
                participation_index=1.0,
            )
            for event_type, event_set in self.events_by_type.items()
        ]

    def _make_length_two_level(self) -> list[_PatternNode]:
        level: list[_PatternNode] = []
        singletons = self.levels[0]
        for first_parent in singletons:
            for second_parent in singletons:
                if second_parent.event_type == first_parent.event_type:
                    continue

                join_result = plane_sweep_follow_join(
                    first_parent.tail_events,
                    second_parent.tail_events,
                    spatial_radius=self.spatial_radius,
                    temporal_window=self.temporal_window,
                )
                if not join_result.tail_events:
                    continue

                participation_ratio = self._participation_ratio(
                    second_parent.event_type,
                    join_result.tail_events,
                )
                participation_index = min(
                    first_parent.participation_index,
                    participation_ratio,
                )
                if not self._passes_threshold(participation_index):
                    continue

                node = _PatternNode(
                    event_type=second_parent.event_type,
                    tail_events=join_result.tail_events,
                    participation_index=participation_index,
                    first_parent=first_parent,
                    second_parent=second_parent,
                )
                first_parent.children.append(node)
                level.append(node)
        return level

    def _generate_next_level(self, previous_level: list[_PatternNode]) -> list[_PatternNode]:
        next_level: list[_PatternNode] = []
        for first_parent in previous_level:
            if first_parent.second_parent is None:
                continue

            for second_parent_child in first_parent.second_parent.children:
                if second_parent_child.event_type == first_parent.event_type:
                    continue

                join_result = plane_sweep_follow_join(
                    first_parent.tail_events,
                    second_parent_child.tail_events,
                    spatial_radius=self.spatial_radius,
                    temporal_window=self.temporal_window,
                )
                if not join_result.tail_events:
                    continue

                participation_ratio = self._participation_ratio(
                    second_parent_child.event_type,
                    join_result.tail_events,
                )
                participation_index = min(
                    first_parent.participation_index,
                    participation_ratio,
                )
                if not self._passes_threshold(participation_index):
                    continue

                node = _PatternNode(
                    event_type=second_parent_child.event_type,
                    tail_events=join_result.tail_events,
                    participation_index=participation_index,
                    first_parent=first_parent,
                    second_parent=second_parent_child,
                )
                first_parent.children.append(node)
                next_level.append(node)
        return next_level

    def _passes_threshold(self, participation_index: float) -> bool:
        if self.threshold_inclusive:
            return participation_index >= self.min_participation_index
        return participation_index > self.min_participation_index

    def _participation_ratio(
        self,
        event_type: str,
        tail_events: tuple[EventInstance, ...],
    ) -> float:
        total_count = self.event_type_counts[event_type]
        if total_count == 0:
            return 0.0
        return len(tail_events) / total_count

    def _to_pattern(
        self,
        node: _PatternNode,
        last_participation_ratio: float | None = None,
    ) -> STBFMPattern:
        if last_participation_ratio is None:
            last_participation_ratio = self._participation_ratio(
                node.event_type,
                node.tail_events,
            )
        return STBFMPattern(
            sequence=node.sequence,
            participation_index=node.participation_index,
            tail_event_count=len(node.tail_events),
            last_participation_ratio=last_participation_ratio,
        )

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


def patterns_to_frame(patterns: Iterable[STBFMPattern]) -> pd.DataFrame:
    """Convert discovered STBFM patterns to an analysis-friendly DataFrame."""

    return pd.DataFrame(
        [
            {
                "sequence": " -> ".join(pattern.sequence),
                "length": pattern.length,
                "participation_index": pattern.participation_index,
                "tail_event_count": pattern.tail_event_count,
                "last_participation_ratio": pattern.last_participation_ratio,
            }
            for pattern in patterns
        ]
    )

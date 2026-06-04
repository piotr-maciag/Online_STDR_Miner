from __future__ import annotations

import colorsys
from io import BytesIO
import json
from math import atan, degrees, log, pi, radians, sinh, sqrt, tan
from pathlib import Path
from time import sleep

import pandas as pd

from spatio_temporal_network import (
    LearningParameters,
    OnlineSTDRMiner,
    SpatioTemporalNetwork,
    initialize_network_from_events,
)


def build_miner_state_at_event(
    events: pd.DataFrame,
    event_number: int,
    grid_shape: tuple[int, int],
    spatial_threshold: float,
    parameters: LearningParameters,
) -> tuple[SpatioTemporalNetwork, OnlineSTDRMiner, pd.DataFrame]:
    """Initialize N and process the stream up to the requested event number."""
    if event_number <= 0:
        raise ValueError("event_number must be a positive integer.")
    if events.empty:
        raise ValueError("events must contain at least one row.")

    sorted_events = events.sort_values("timestamp").reset_index(drop=True)
    processed_events = sorted_events.head(event_number).copy()
    if processed_events.empty:
        raise ValueError("No events selected for processing.")

    network = initialize_network_from_events(
        events=sorted_events,
        grid_shape=grid_shape,
        spatial_threshold=spatial_threshold,
    )
    miner = OnlineSTDRMiner(network, parameters)
    for event in processed_events.itertuples(index=False):
        miner.process_event(pd.Series(event._asdict()), extract_rules=False)

    return network, miner, processed_events


def visualize_significant_synapse_cube(
    events: pd.DataFrame,
    network: SpatioTemporalNetwork,
    miner: OnlineSTDRMiner,
    processed_events: pd.DataFrame,
    title: str | None = None,
    output_html: Path | str | None = None,
    show_all_event_points: bool = False,
    show_basemap: bool = True,
    basemap_zoom: int = 12,
    basemap_resolution: tuple[int, int] = (120, 120),
    basemap_trace_data: dict | None = None,
    max_synapses: int | None = None,
):
    """Create an interactive cube visualization of significant synapses.

    The bottom plane shows event coordinates. Neurons are placed at grid-cell
    centers, one event type per z-layer. Significant synapses are rendered as
    directed lines whose color and width reflect ``weight / w_max``.
    """
    go = _import_plotly()

    if processed_events.empty:
        raise ValueError("processed_events must contain at least one row.")
    if max_synapses is not None and max_synapses <= 0:
        raise ValueError("max_synapses must be positive or None.")

    z_bottom = -1.0
    z_by_event_type = {
        event_type: index
        for index, event_type in enumerate(network.event_types)
    }
    event_type_colors = _event_type_color_map(network.event_types)
    z_top = max(z_by_event_type.values(), default=0)

    fig = go.Figure()
    _add_bottom_event_map(
        fig=fig,
        go=go,
        events=events,
        processed_events=processed_events,
        z_bottom=z_bottom,
        show_all_event_points=show_all_event_points,
        show_basemap=show_basemap,
        basemap_zoom=basemap_zoom,
        basemap_resolution=basemap_resolution,
        basemap_trace_data=basemap_trace_data,
        event_type_colors=event_type_colors,
    )
    _add_grid_wireframe(fig=fig, go=go, network=network, z_bottom=z_bottom, z_top=z_top)
    significant_synapses = _significant_synapse_items(network, miner, max_synapses)
    _add_neuron_layers(
        fig=fig,
        go=go,
        network=network,
        z_by_event_type=z_by_event_type,
        significant_synapses=significant_synapses,
        event_type_colors=event_type_colors,
    )
    _add_significant_synapses(
        fig=fig,
        go=go,
        network=network,
        miner=miner,
        z_by_event_type=z_by_event_type,
        significant_synapses=significant_synapses,
    )
    processed_time_span = _processed_time_span_label(processed_events)
    figure_title = title or (
        f"Significant synapse cube after {miner.processed_events} events "
        f"({processed_time_span}, theta={miner.parameters.theta})"
    )
    fig.update_layout(
        title=figure_title,
        scene={
            "xaxis": {
                "title": "Longitude",
                "range": [network.bounds.min_longitude, network.bounds.max_longitude],
                "autorange": False,
            },
            "yaxis": {
                "title": "Latitude",
                "range": [network.bounds.min_latitude, network.bounds.max_latitude],
                "autorange": False,
            },
            "zaxis": {
                "title": "",
                "tickmode": "array",
                "tickvals": list(z_by_event_type.values()),
                "ticktext": list(z_by_event_type.keys()),
                "showticklabels": True,
            },
            "aspectmode": "cube",
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        },
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 42, "b": 0},
        autosize=True,
        height=900,
    )

    if output_html is not None:
        output_html = Path(output_html)
        output_html.parent.mkdir(parents=True, exist_ok=True)
        synapse_rows = _significant_synapse_rows(network, miner, max_synapses)
        _write_filterable_html(
            fig,
            output_html,
            network.event_types,
            event_type_colors,
            synapse_rows,
            (
                network.bounds.min_longitude,
                network.bounds.max_longitude,
            ),
            (
                network.bounds.min_latitude,
                network.bounds.max_latitude,
            ),
        )

    return fig


def build_and_visualize_at_event(
    events: pd.DataFrame,
    event_number: int,
    grid_shape: tuple[int, int],
    spatial_threshold: float,
    parameters: LearningParameters,
    title: str | None = None,
    output_html: Path | str | None = None,
    show_all_event_points: bool = False,
    show_basemap: bool = True,
    basemap_zoom: int = 12,
    basemap_resolution: tuple[int, int] = (120, 120),
    basemap_trace_data: dict | None = None,
    max_synapses: int | None = None,
):
    """Convenience wrapper for processing and visualizing a stream prefix."""
    network, miner, processed_events = build_miner_state_at_event(
        events=events,
        event_number=event_number,
        grid_shape=grid_shape,
        spatial_threshold=spatial_threshold,
        parameters=parameters,
    )
    fig = visualize_significant_synapse_cube(
        events=events,
        network=network,
        miner=miner,
        processed_events=processed_events,
        title=title,
        output_html=output_html,
        show_all_event_points=show_all_event_points,
        show_basemap=show_basemap,
        basemap_zoom=basemap_zoom,
        basemap_resolution=basemap_resolution,
        basemap_trace_data=basemap_trace_data,
        max_synapses=max_synapses,
    )
    return fig, network, miner, processed_events


def _import_plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Plotly is required for cube visualization. Install project "
            "requirements, including plotly, before running this notebook."
        ) from exc
    return go


def _processed_time_span_label(processed_events: pd.DataFrame) -> str:
    if "Occurrence_time" in processed_events.columns:
        first_time = processed_events["Occurrence_time"].iloc[0]
        last_time = processed_events["Occurrence_time"].iloc[-1]
        return f"time span: {first_time} - {last_time}"

    first_timestamp = float(processed_events["timestamp"].iloc[0])
    last_timestamp = float(processed_events["timestamp"].iloc[-1])
    return f"time span: t={first_timestamp:.2f} - t={last_timestamp:.2f}"


def _write_filterable_html(
    fig,
    output_html: Path,
    event_types: tuple[str, ...],
    event_type_colors: dict[str, str],
    synapse_rows: list[dict],
    longitude_range: tuple[float, float],
    latitude_range: tuple[float, float],
) -> None:
    plot_div = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        div_id="rule-cube",
        config={"responsive": True},
    )
    checkbox_html = "\n".join(
        (
            '<label class="event-filter-item">'
            f'<input type="checkbox" class="event-filter" value="{_html_escape(event_type)}" checked> '
            '<span class="event-filter-swatch" '
            f'style="background:{_html_escape(event_type_colors[event_type])}"></span>'
            f'{_html_escape(event_type)}'
            "</label>"
        )
        for event_type in event_types
    )
    event_types_json = json.dumps(list(event_types))
    synapse_rows_json = json.dumps(synapse_rows)
    longitude_range_json = json.dumps(list(longitude_range))
    latitude_range_json = json.dumps(list(latitude_range))
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Rule Cube</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #1f2933;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(150px, 190px) minmax(0, 1fr) minmax(300px, 420px);
      grid-template-areas:
        "filters figure synapses";
      height: 100vh;
      width: 100vw;
    }}
    .filters {{
      border-right: 1px solid #d6d9de;
      padding: 10px;
      overflow: auto;
      background: #f7f8fa;
      grid-area: filters;
    }}
    .filters h2 {{
      margin: 0 0 12px;
      font-size: 14px;
      font-weight: 650;
    }}
    .filter-actions {{
      display: flex;
      gap: 8px;
      margin-bottom: 10px;
    }}
    .filter-actions button {{
      border: 1px solid #b8bec8;
      background: #fff;
      border-radius: 4px;
      padding: 3px 7px;
      cursor: pointer;
    }}
    .top-percent-control {{
      border-bottom: 1px solid #dfe3e8;
      margin-bottom: 12px;
      padding-bottom: 12px;
      font-size: 12px;
    }}
    .top-percent-control label {{
      display: block;
      font-weight: 650;
      margin-bottom: 6px;
    }}
    .top-percent-control input {{
      width: 72px;
      box-sizing: border-box;
      border: 1px solid #b8bec8;
      border-radius: 4px;
      padding: 3px 6px;
      font: inherit;
    }}
    .event-filter-item {{
      display: flex;
      align-items: center;
      gap: 6px;
      margin: 5px 0;
      font-size: 12px;
      line-height: 1.25;
    }}
    .event-filter-swatch {{
      display: inline-block;
      width: 10px;
      height: 10px;
      border: 1px solid rgba(0,0,0,0.22);
      flex: 0 0 auto;
    }}
    .figure {{
      grid-area: figure;
      min-width: 0;
      min-height: 0;
    }}
    .synapse-panel {{
      border-left: 1px solid #d6d9de;
      padding: 10px;
      overflow: auto;
      background: #ffffff;
      grid-area: synapses;
    }}
    .synapse-panel h2 {{
      margin: 0 0 8px;
      font-size: 14px;
      font-weight: 650;
    }}
    .synapse-count {{
      margin-bottom: 10px;
      color: #52606d;
      font-size: 12px;
    }}
    .synapse-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    .synapse-table th,
    .synapse-table td {{
      border-bottom: 1px solid #e6e8eb;
      padding: 5px 4px;
      text-align: left;
      vertical-align: top;
    }}
    .synapse-table th {{
      position: sticky;
      top: 0;
      background: #ffffff;
      z-index: 1;
      font-weight: 650;
    }}
    .synapse-table td.number {{
      text-align: right;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }}
    #rule-cube {{
      width: 100%;
      height: 100%;
    }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="filters">
      <div class="top-percent-control">
        <label for="top-percent">
          Top synapses (%)
        </label>
        <input type="number" id="top-percent" min="1" max="100" step="1" value="10">
      </div>
      <h2>Shown Event Types</h2>
      <div class="filter-actions">
        <button type="button" id="select-all">All</button>
        <button type="button" id="select-none">None</button>
      </div>
      {checkbox_html}
    </aside>
    <main class="figure">
      {plot_div}
    </main>
    <aside class="synapse-panel">
      <h2>Most Significant Synapses</h2>
      <div class="synapse-count" id="synapse-count"></div>
      <table class="synapse-table">
        <thead>
          <tr>
            <th>Synapse</th>
            <th>Cells</th>
            <th>Weight</th>
            <th>Ratio</th>
          </tr>
        </thead>
        <tbody id="synapse-table-body"></tbody>
      </table>
    </aside>
  </div>
  <script>
    const EVENT_TYPES = {event_types_json};
    const SYNAPSE_ROWS = {synapse_rows_json};
    const LON_RANGE = {longitude_range_json};
    const LAT_RANGE = {latitude_range_json};
    const graph = document.getElementById("rule-cube");

    function selectedEventTypes() {{
      return new Set(
        Array.from(document.querySelectorAll(".event-filter"))
          .filter((checkbox) => checkbox.checked)
          .map((checkbox) => checkbox.value)
      );
    }}

    function selectedEventTypeOrder(selected) {{
      return EVENT_TYPES.filter((eventType) => selected.has(eventType));
    }}

    function selectedTopPercent() {{
      const value = Number(document.getElementById("top-percent").value);
      if (!Number.isFinite(value)) {{
        return 10;
      }}
      return Math.min(100, Math.max(1, value));
    }}

    function topCount(total, percent) {{
      if (!total) {{
        return 0;
      }}
      return Math.max(1, Math.ceil(total * percent / 100));
    }}

    function traceVisible(trace, selected, topPercent) {{
      const meta = trace.meta || {{}};
      const traceEventTypes = meta.event_types || [];
      if (meta.role === "significant_synapse" && meta.top_rank !== undefined) {{
        return meta.top_rank < topCount(SYNAPSE_ROWS.length, topPercent)
          && traceEventTypes.every((eventType) => selected.has(eventType));
      }}
      if (meta.role === "neuron") {{
        if (!traceEventTypes.every((eventType) => selected.has(eventType))) {{
          return false;
        }}
        const synapseLimit = topCount(SYNAPSE_ROWS.length, topPercent);
        return (meta.connected_synapses || []).some((synapse) =>
          synapse.top_rank < synapseLimit
          && synapse.event_types.every((eventType) => selected.has(eventType))
        );
      }}
      if (!traceEventTypes.length) {{
        return true;
      }}
      return traceEventTypes.every((eventType) => selected.has(eventType));
    }}

    function originalTraceValue(trace, key) {{
      if (!trace._original) {{
        trace._original = {{
          x: trace.x ? Array.from(trace.x) : undefined,
          y: trace.y ? Array.from(trace.y) : undefined,
          z: trace.z ? Array.from(trace.z) : undefined,
          u: trace.u ? Array.from(trace.u) : undefined,
          v: trace.v ? Array.from(trace.v) : undefined,
          w: trace.w ? Array.from(trace.w) : undefined,
        }};
      }}
      return trace._original[key];
    }}

    function compactTraceZ(trace, selectedOrder) {{
      const meta = trace.meta || {{}};
      const traceEventTypes = meta.event_types || [];
      if (meta.fixed_z) {{
        return originalTraceValue(trace, "z");
      }}
      const zMap = new Map(selectedOrder.map((eventType, index) => [eventType, index]));
      if (!traceEventTypes.length) {{
        return originalTraceValue(trace, "z");
      }}
      if (traceEventTypes.length === 1) {{
        if (!zMap.has(traceEventTypes[0])) {{
          return originalTraceValue(trace, "z");
        }}
        const layer = zMap.get(traceEventTypes[0]);
        return originalTraceValue(trace, "z").map(() => layer);
      }}
      if (originalTraceValue(trace, "z").length === traceEventTypes.length) {{
        if (!traceEventTypes.every((eventType) => zMap.has(eventType))) {{
          return originalTraceValue(trace, "z");
        }}
        return traceEventTypes.map((eventType) => zMap.get(eventType));
      }}
      if (!zMap.has(traceEventTypes[0]) || !zMap.has(traceEventTypes[1])) {{
        return originalTraceValue(trace, "z");
      }}
      const startLayer = zMap.get(traceEventTypes[0]);
      const endLayer = zMap.get(traceEventTypes[1]);
      if (trace.type === "cone") {{
        return [(startLayer + 0.88 * (endLayer - startLayer))];
      }}
      return [startLayer, endLayer];
    }}

    function compactConeW(trace, selectedOrder) {{
      const meta = trace.meta || {{}};
      const traceEventTypes = meta.event_types || [];
      if (trace.type !== "cone" || traceEventTypes.length < 2) {{
        return originalTraceValue(trace, "w");
      }}
      const zMap = new Map(selectedOrder.map((eventType, index) => [eventType, index]));
      if (!zMap.has(traceEventTypes[0]) || !zMap.has(traceEventTypes[1])) {{
        return originalTraceValue(trace, "w");
      }}
      const dz = zMap.get(traceEventTypes[1]) - zMap.get(traceEventTypes[0]);
      const originalU = originalTraceValue(trace, "u")[0];
      const originalV = originalTraceValue(trace, "v")[0];
      const norm = Math.sqrt(originalU * originalU + originalV * originalV + dz * dz);
      return [norm === 0 ? 0 : dz / norm];
    }}

    function applyEventTypeFilter() {{
      const selected = selectedEventTypes();
      const selectedOrder = selectedEventTypeOrder(selected);
      const topPercent = selectedTopPercent();
      const visible = graph.data.map((trace) => traceVisible(trace, selected, topPercent));
      const updatedZ = graph.data.map((trace) => compactTraceZ(trace, selectedOrder));
      const updatedW = graph.data.map((trace) => compactConeW(trace, selectedOrder));
      Plotly.restyle(graph, {{visible: visible, z: updatedZ, w: updatedW}});
      Plotly.relayout(graph, {{
        "scene.zaxis.tickmode": "array",
        "scene.zaxis.tickvals": selectedOrder.map((_, index) => index),
        "scene.zaxis.ticktext": selectedOrder,
        "scene.zaxis.range": [-1.15, Math.max(0, selectedOrder.length - 1) + 0.35],
        "scene.xaxis.range": LON_RANGE,
        "scene.xaxis.autorange": false,
        "scene.yaxis.range": LAT_RANGE,
        "scene.yaxis.autorange": false,
      }});
      renderSynapseTable(selected);
    }}

    function renderSynapseTable(selected) {{
      const body = document.getElementById("synapse-table-body");
      const topPercent = selectedTopPercent();
      const synapseLimit = topCount(SYNAPSE_ROWS.length, topPercent);
      const visibleRows = SYNAPSE_ROWS.filter((row) =>
        row.top_rank < synapseLimit
        && selected.has(row.presynaptic_event_type)
        && selected.has(row.postsynaptic_event_type)
      );
      document.getElementById("synapse-count").textContent =
        `${{visibleRows.length}} shown of top ${{synapseLimit}} / ${{SYNAPSE_ROWS.length}} saved`;
      body.innerHTML = visibleRows.map((row) => `
        <tr>
          <td>${{escapeHtml(row.presynaptic_event_type)}} &rarr; ${{escapeHtml(row.postsynaptic_event_type)}}</td>
          <td>${{row.presynaptic_cell_id}} &rarr; ${{row.postsynaptic_cell_id}}</td>
          <td class="number">${{row.weight.toFixed(6)}}</td>
          <td class="number">${{row.weight_ratio.toFixed(6)}}</td>
        </tr>
      `).join("");
    }}

    function escapeHtml(value) {{
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }}

    document.querySelectorAll(".event-filter").forEach((checkbox) => {{
      checkbox.addEventListener("change", applyEventTypeFilter);
    }});
    document.getElementById("select-all").addEventListener("click", () => {{
      document.querySelectorAll(".event-filter").forEach((checkbox) => checkbox.checked = true);
      applyEventTypeFilter();
    }});
    document.getElementById("select-none").addEventListener("click", () => {{
      document.querySelectorAll(".event-filter").forEach((checkbox) => checkbox.checked = false);
      applyEventTypeFilter();
    }});
    document.getElementById("top-percent").addEventListener("input", () => {{
      applyEventTypeFilter();
    }});
    applyEventTypeFilter();
    window.addEventListener("resize", () => Plotly.Plots.resize(graph));
    Plotly.Plots.resize(graph);
  </script>
</body>
</html>
"""
    output_html.write_text(html, encoding="utf-8")


def _html_escape(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _event_type_color_map(event_types: tuple[str, ...]) -> dict[str, str]:
    return {
        event_type: _event_type_color(index, len(event_types))
        for index, event_type in enumerate(event_types)
    }


def _event_type_color(index: int, event_type_count: int) -> str:
    if event_type_count <= 0:
        event_type_count = 1
    hue = (index * 0.618033988749895) % 1.0
    saturation = 0.58 + 0.18 * ((index // event_type_count) % 2)
    value = 0.78 + 0.12 * ((index % 3) / 2)
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def _significant_synapse_items(
    network: SpatioTemporalNetwork,
    miner: OnlineSTDRMiner,
    max_synapses: int | None,
) -> list[tuple[float, int, int]]:
    significant_synapses = []
    for presynaptic_id, postsynaptic_id in miner.significant_synapses:
        state = network.get_synapse_state(presynaptic_id, postsynaptic_id)
        significant_synapses.append(
            (state.weight / miner.parameters.max_weight, presynaptic_id, postsynaptic_id)
        )
    significant_synapses.sort(reverse=True)
    if max_synapses is not None:
        significant_synapses = significant_synapses[:max_synapses]
    return significant_synapses


def _significant_synapse_rows(
    network: SpatioTemporalNetwork,
    miner: OnlineSTDRMiner,
    max_synapses: int | None,
) -> list[dict]:
    rows = []
    for presynaptic_id, postsynaptic_id in miner.significant_synapses:
        presynaptic = network.neurons[presynaptic_id]
        postsynaptic = network.neurons[postsynaptic_id]
        state = network.get_synapse_state(presynaptic_id, postsynaptic_id)
        rows.append(
            {
                "presynaptic_event_type": presynaptic.event_type,
                "postsynaptic_event_type": postsynaptic.event_type,
                "presynaptic_cell_id": presynaptic.cell_id,
                "postsynaptic_cell_id": postsynaptic.cell_id,
                "weight": state.weight,
                "weight_ratio": state.weight / miner.parameters.max_weight,
            }
        )
    rows.sort(key=lambda row: row["weight_ratio"], reverse=True)
    if max_synapses is not None:
        rows = rows[:max_synapses]
    for top_rank, row in enumerate(rows):
        row["top_rank"] = top_rank
    return rows


def _add_bottom_event_map(
    fig,
    go,
    events: pd.DataFrame,
    processed_events: pd.DataFrame,
    z_bottom: float,
    show_all_event_points: bool,
    show_basemap: bool,
    basemap_zoom: int,
    basemap_resolution: tuple[int, int],
    basemap_trace_data: dict | None,
    event_type_colors: dict[str, str],
) -> None:
    if show_basemap:
        try:
            if basemap_trace_data is None:
                basemap_trace_data = prepare_osm_basemap_trace(
                    events=events,
                    zoom=basemap_zoom,
                    resolution=basemap_resolution,
                    z_bottom=z_bottom,
                )
            _add_osm_basemap_trace(fig=fig, go=go, trace_data=basemap_trace_data)
        except Exception as exc:
            fig.add_annotation(
                text=f"Basemap unavailable: {exc}",
                x=0.01,
                y=0.01,
                xref="paper",
                yref="paper",
                showarrow=False,
            )

    if show_all_event_points:
        for event_type, event_group in events.groupby("Event_type", sort=False):
            fig.add_trace(
                go.Scatter3d(
                    x=event_group["longitude"],
                    y=event_group["latitude"],
                    z=[z_bottom] * len(event_group),
                    mode="markers",
                    marker={"size": 2, "color": "rgba(120, 120, 120, 0.18)"},
                    name=f"All coordinates: {event_type}",
                    showlegend=False,
                    meta={"event_types": [str(event_type)], "fixed_z": True},
                    hovertemplate="lon=%{x}<br>lat=%{y}<extra></extra>",
                )
            )

    for event_type, event_group in processed_events.groupby("Event_type", sort=False):
        fig.add_trace(
            go.Scatter3d(
                x=event_group["longitude"],
                y=event_group["latitude"],
                z=[z_bottom] * len(event_group),
                mode="markers",
                marker={
                    "size": 3,
                    "color": event_type_colors[str(event_type)],
                    "opacity": 0.75,
                },
                text=event_group["Event_type"],
                name=f"Processed events: {event_type}",
                showlegend=False,
                meta={"event_types": [str(event_type)], "fixed_z": True},
                hovertemplate=(
                    "type=%{text}<br>lon=%{x}<br>lat=%{y}<extra></extra>"
                ),
            )
        )


def _add_grid_wireframe(
    fig,
    go,
    network: SpatioTemporalNetwork,
    z_bottom: float,
    z_top: float,
) -> None:
    bounds = network.bounds
    rows, columns = network.grid_shape
    lon_step = (bounds.max_longitude - bounds.min_longitude) / columns
    lat_step = (bounds.max_latitude - bounds.min_latitude) / rows

    line_color = "rgba(80, 80, 80, 0.35)"
    for column in range(columns + 1):
        lon = bounds.min_longitude + column * lon_step
        fig.add_trace(
            go.Scatter3d(
                x=[lon, lon],
                y=[bounds.min_latitude, bounds.max_latitude],
                z=[z_bottom, z_bottom],
                mode="lines",
                line={"color": line_color, "width": 2},
                showlegend=False,
                hoverinfo="skip",
            )
        )
    for row in range(rows + 1):
        lat = bounds.min_latitude + row * lat_step
        fig.add_trace(
            go.Scatter3d(
                x=[bounds.min_longitude, bounds.max_longitude],
                y=[lat, lat],
                z=[z_bottom, z_bottom],
                mode="lines",
                line={"color": line_color, "width": 2},
                showlegend=False,
                hoverinfo="skip",
            )
        )

    for lon in (bounds.min_longitude, bounds.max_longitude):
        for lat in (bounds.min_latitude, bounds.max_latitude):
            fig.add_trace(
                go.Scatter3d(
                    x=[lon, lon],
                    y=[lat, lat],
                    z=[z_bottom, z_top],
                    mode="lines",
                    line={"color": "rgba(80, 80, 80, 0.2)", "width": 2},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )


def _add_neuron_layers(
    fig,
    go,
    network: SpatioTemporalNetwork,
    z_by_event_type: dict[str, int],
    significant_synapses: list[tuple[float, int, int]],
    event_type_colors: dict[str, str],
) -> None:
    synapses_by_neuron_id: dict[int, list[dict]] = {}
    for top_rank, (_, presynaptic_id, postsynaptic_id) in enumerate(significant_synapses):
        presynaptic = network.neurons[presynaptic_id]
        postsynaptic = network.neurons[postsynaptic_id]
        synapse_meta = {
            "top_rank": top_rank,
            "event_types": [presynaptic.event_type, postsynaptic.event_type],
        }
        synapses_by_neuron_id.setdefault(presynaptic_id, []).append(synapse_meta)
        synapses_by_neuron_id.setdefault(postsynaptic_id, []).append(synapse_meta)

    if not synapses_by_neuron_id:
        return

    for neuron_id, connected_synapses in synapses_by_neuron_id.items():
        neuron = network.neurons[neuron_id]
        fig.add_trace(
            go.Scatter3d(
                x=[neuron.longitude],
                y=[neuron.latitude],
                z=[z_by_event_type[neuron.event_type]],
                mode="markers",
                marker={
                    "size": 4,
                    "color": event_type_colors[neuron.event_type],
                    "opacity": 0.78,
                },
                text=[(
                    f"neuron={neuron.neuron_id}<br>"
                    f"type={neuron.event_type}<br>"
                    f"cell={neuron.cell_id}"
                )],
                name=f"Involved neuron: {neuron.event_type}",
                showlegend=False,
                meta={
                    "role": "neuron",
                    "event_types": [neuron.event_type],
                    "connected_synapses": connected_synapses,
                },
                hovertemplate="%{text}<br>lon=%{x}<br>lat=%{y}<extra></extra>",
            )
        )


def _add_significant_synapses(
    fig,
    go,
    network: SpatioTemporalNetwork,
    miner: OnlineSTDRMiner,
    z_by_event_type: dict[str, int],
    significant_synapses: list[tuple[float, int, int]],
) -> None:
    if significant_synapses:
        weight_ratios = [item[0] for item in significant_synapses]
        min_weight_ratio = min(weight_ratios)
        max_weight_ratio = max(weight_ratios)
        _add_synapse_colorbar(fig=fig, go=go, weight_ratios=weight_ratios)
    else:
        min_weight_ratio = 0.0
        max_weight_ratio = 1.0

    for top_rank, (weight_ratio, presynaptic_id, postsynaptic_id) in enumerate(significant_synapses):
        presynaptic = network.neurons[presynaptic_id]
        postsynaptic = network.neurons[postsynaptic_id]
        color = _weight_ratio_to_color(weight_ratio, min_weight_ratio, max_weight_ratio)
        fig.add_trace(
            go.Scatter3d(
                x=[presynaptic.longitude, postsynaptic.longitude],
                y=[presynaptic.latitude, postsynaptic.latitude],
                z=[
                    z_by_event_type[presynaptic.event_type],
                    z_by_event_type[postsynaptic.event_type],
                ],
                mode="lines",
                line={"color": color, "width": 4},
                name="Significant synapse",
                legendgroup="significant_synapse",
                showlegend=False,
                meta={
                    "role": "significant_synapse",
                    "top_rank": top_rank,
                    "event_types": [
                        presynaptic.event_type,
                        postsynaptic.event_type,
                    ]
                },
                text=[
                    (
                        f"{presynaptic.event_type} -> {postsynaptic.event_type}<br>"
                        f"weight_ratio={weight_ratio:.6f}<br>"
                        f"theta={miner.parameters.theta:.6f}"
                    ),
                    (
                        f"{presynaptic.event_type} -> {postsynaptic.event_type}<br>"
                        f"weight_ratio={weight_ratio:.6f}<br>"
                        f"theta={miner.parameters.theta:.6f}"
                    ),
                ],
                hovertemplate="%{text}<extra></extra>",
            )
        )
        _add_synapse_arrowhead(
            fig=fig,
            go=go,
            presynaptic=presynaptic,
            postsynaptic=postsynaptic,
            z_by_event_type=z_by_event_type,
            color=color,
            weight_ratio=weight_ratio,
            theta=miner.parameters.theta,
            top_rank=top_rank,
        )


def _add_synapse_arrowhead(
    fig,
    go,
    presynaptic,
    postsynaptic,
    z_by_event_type: dict[str, int],
    color: str,
    weight_ratio: float,
    theta: float,
    top_rank: int,
) -> None:
    start = (
        presynaptic.longitude,
        presynaptic.latitude,
        float(z_by_event_type[presynaptic.event_type]),
    )
    end = (
        postsynaptic.longitude,
        postsynaptic.latitude,
        float(z_by_event_type[postsynaptic.event_type]),
    )
    direction = (
        end[0] - start[0],
        end[1] - start[1],
        end[2] - start[2],
    )
    direction_length = sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
    if direction_length == 0:
        return

    arrow_position = (
        start[0] + 0.88 * direction[0],
        start[1] + 0.88 * direction[1],
        start[2] + 0.88 * direction[2],
    )
    hover_text = (
        f"{presynaptic.event_type} -> {postsynaptic.event_type}<br>"
        f"weight_ratio={weight_ratio:.6f}<br>"
        f"theta={theta:.6f}"
    )
    fig.add_trace(
        go.Cone(
            x=[arrow_position[0]],
            y=[arrow_position[1]],
            z=[arrow_position[2]],
            u=[direction[0] / direction_length],
            v=[direction[1] / direction_length],
            w=[direction[2] / direction_length],
            sizemode="absolute",
            sizeref=0.18,
            anchor="tip",
            colorscale=[[0.0, color], [1.0, color]],
            showscale=False,
            name="Synapse direction",
            showlegend=False,
            meta={
                "role": "significant_synapse",
                "top_rank": top_rank,
                "event_types": [
                    presynaptic.event_type,
                    postsynaptic.event_type,
                ]
            },
            text=[hover_text],
            hovertemplate="%{text}<extra></extra>",
        )
    )


def _add_synapse_colorbar(fig, go, weight_ratios: list[float]) -> None:
    min_weight_ratio = min(weight_ratios)
    max_weight_ratio = max(weight_ratios)
    if max_weight_ratio <= min_weight_ratio:
        max_weight_ratio = min_weight_ratio + 1e-12
    fig.add_trace(
        go.Scatter3d(
            x=[None] * len(weight_ratios),
            y=[None] * len(weight_ratios),
            z=[None] * len(weight_ratios),
            mode="markers",
            marker={
                "size": 0.1,
                "color": weight_ratios,
                "colorscale": "Viridis",
                "cmin": min_weight_ratio,
                "cmax": max_weight_ratio,
                "showscale": True,
                "colorbar": {
                    "title": "weight / w_max",
                    "thickness": 14,
                    "len": 0.72,
                },
            },
            hoverinfo="skip",
            showlegend=False,
            name="Synapse strength scale",
            meta={"role": "significant_synapse"},
        )
    )


def _significant_synapse_neuron_ids(miner: OnlineSTDRMiner) -> set[int]:
    neuron_ids = set()
    for presynaptic_id, postsynaptic_id in miner.significant_synapses:
        neuron_ids.add(presynaptic_id)
        neuron_ids.add(postsynaptic_id)
    return neuron_ids


def prepare_osm_basemap_trace(
    events: pd.DataFrame,
    zoom: int = 12,
    resolution: tuple[int, int] = (120, 120),
    z_bottom: float = -1.0,
) -> dict:
    """Fetch and rasterize the OpenStreetMap basemap once for reuse."""
    image, lon_min, lon_max, lat_min, lat_max = _fetch_osm_mosaic(events, zoom)
    width, height = resolution
    image = image.resize((width, height))
    rgb_points = list(image.convert("RGB").getdata())

    xs = []
    ys = []
    zs = []
    colors = []
    for row in range(height):
        lat = lat_max - (lat_max - lat_min) * row / max(1, height - 1)
        for column in range(width):
            lon = lon_min + (lon_max - lon_min) * column / max(1, width - 1)
            red, green, blue = rgb_points[row * width + column]
            xs.append(lon)
            ys.append(lat)
            zs.append(z_bottom - 0.02)
            colors.append(f"rgb({red},{green},{blue})")

    return {
        "x": xs,
        "y": ys,
        "z": zs,
        "colors": colors,
    }


def _add_osm_basemap_trace(
    fig,
    go,
    trace_data: dict,
) -> None:
    fig.add_trace(
        go.Scatter3d(
            x=trace_data["x"],
            y=trace_data["y"],
            z=trace_data["z"],
            mode="markers",
            marker={"size": 2.5, "color": trace_data["colors"], "opacity": 0.95},
            name="OpenStreetMap basemap",
            hoverinfo="skip",
        )
    )


def _fetch_osm_mosaic(events: pd.DataFrame, zoom: int):
    requests, Image = _import_map_dependencies()
    lon_min = float(events["longitude"].min())
    lon_max = float(events["longitude"].max())
    lat_min = float(events["latitude"].min())
    lat_max = float(events["latitude"].max())

    x_min, y_max = _lonlat_to_tile(lon_min, lat_min, zoom)
    x_max, y_min = _lonlat_to_tile(lon_max, lat_max, zoom)
    tile_size = 256
    mosaic = Image.new(
        "RGB",
        ((x_max - x_min + 1) * tile_size, (y_max - y_min + 1) * tile_size),
        "white",
    )

    headers = {"User-Agent": "SNNs-ST-Rules visualization"}
    for tile_x in range(x_min, x_max + 1):
        for tile_y in range(y_min, y_max + 1):
            url = f"https://tile.openstreetmap.org/{zoom}/{tile_x}/{tile_y}.png"
            for attempt in range(2):
                try:
                    response = requests.get(url, headers=headers, timeout=8)
                    response.raise_for_status()
                    break
                except requests.RequestException:
                    if attempt == 1:
                        raise
                    sleep(1.0)
            tile = Image.open(BytesIO(response.content)).convert("RGB")
            mosaic.paste(tile, ((tile_x - x_min) * tile_size, (tile_y - y_min) * tile_size))

    top_left_lon, top_left_lat = _tile_to_lonlat(x_min, y_min, zoom)
    bottom_right_lon, bottom_right_lat = _tile_to_lonlat(x_max + 1, y_max + 1, zoom)
    return mosaic, top_left_lon, bottom_right_lon, bottom_right_lat, top_left_lat


def _import_map_dependencies():
    try:
        import requests
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "OpenStreetMap basemap rendering requires requests and pillow. "
            "Install project requirements before running this notebook."
        ) from exc
    return requests, Image


def _lonlat_to_tile(longitude: float, latitude: float, zoom: int) -> tuple[int, int]:
    lat_rad = radians(latitude)
    n = 2**zoom
    x_tile = int((longitude + 180.0) / 360.0 * n)
    y_tile = int((1.0 - log(tan(lat_rad) + 1.0 / cos_lat(lat_rad)) / pi) / 2.0 * n)
    return x_tile, y_tile


def _tile_to_lonlat(x_tile: int, y_tile: int, zoom: int) -> tuple[float, float]:
    n = 2**zoom
    longitude = x_tile / n * 360.0 - 180.0
    latitude = degrees(atan(sinh(pi * (1.0 - 2.0 * y_tile / n))))
    return longitude, latitude


def cos_lat(lat_rad: float) -> float:
    from math import cos

    return cos(lat_rad)


def _weight_ratio_to_color(
    weight_ratio: float,
    min_weight_ratio: float,
    max_weight_ratio: float,
) -> str:
    if max_weight_ratio <= min_weight_ratio:
        scale = 1.0
    else:
        scale = (weight_ratio - min_weight_ratio) / (max_weight_ratio - min_weight_ratio)
    scale = min(1.0, max(0.0, scale))
    viridis = [
        (68, 1, 84),
        (72, 40, 120),
        (62, 74, 137),
        (49, 104, 142),
        (38, 130, 142),
        (31, 158, 137),
        (53, 183, 121),
        (109, 205, 89),
        (180, 222, 44),
        (253, 231, 37),
    ]
    index = min(len(viridis) - 1, int(scale * (len(viridis) - 1)))
    red, green, blue = viridis[index]
    return f"rgb({red}, {green}, {blue})"

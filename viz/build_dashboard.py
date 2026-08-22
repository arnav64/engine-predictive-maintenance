"""
Generates docs/dashboard.html -- an interactive Plotly dashboard of the
engine predictive-maintenance pipeline's real results.

Every figure is built from actual pipeline output, not illustrative data:
  results/rul_predictions_DS02.csv         (RUL model test predictions)
  data/processed/features_DS02.parquet     (ground-truth RUL trajectories)
  results/sdr_events_by_aircraft.csv       (FAA SDR failure patterns)
  results/sdr_annual_trend.csv             (FAA SDR annual trend)
  results/impact_summary.csv               (BTS carrier impact estimate)
  results/sensitivity_grid.csv             (assumption sensitivity sweep)

Usage:
    python viz/build_dashboard.py
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
PROC_DIR    = ROOT / "data" / "processed"
DOCS_DIR    = ROOT / "docs"
OUTPUT_HTML = DOCS_DIR / "dashboard.html"

BLUE       = "#2563eb"
BLUE_DK    = "#1d4ed8"
INK        = "#0f172a"
INK2       = "#475569"
INK3       = "#94a3b8"
BORDER     = "#e2e8f0"
GREEN      = "#15803d"

CATEGORICAL = ["#2563eb", "#0891b2", "#15803d", "#ea580c", "#7c3aed",
               "#dc2626", "#0369a1", "#b45309", "#16a34a"]

FONT = dict(family='"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
            color=INK2, size=12)

BASE_LAYOUT = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=FONT,
    hoverlabel=dict(bgcolor="#ffffff", bordercolor=BORDER,
                     font=dict(family=FONT["family"], size=12, color=INK)),
)
DEFAULT_MARGIN = dict(l=50, r=20, t=10, b=45)
DEFAULT_LEGEND = dict(font=dict(size=11))


def fig_rul_scatter() -> go.Figure:
    df = pd.read_csv(RESULTS_DIR / "rul_predictions_DS02.csv")
    traces = []
    for i, unit in enumerate(sorted(df["unit"].unique())):
        sub = df[df["unit"] == unit]
        traces.append(go.Scatter(
            x=sub["RUL"], y=sub["predicted_RUL"], mode="markers",
            name=f"Engine {int(unit)}",
            marker=dict(size=6, color=CATEGORICAL[i % len(CATEGORICAL)], opacity=0.65),
            hovertemplate="Actual RUL: %{x}<br>Predicted: %{y:.1f}<extra>Engine "
                          + str(int(unit)) + "</extra>",
        ))
    lims = [0, max(df["RUL"].max(), df["predicted_RUL"].max()) + 5]
    traces.append(go.Scatter(x=lims, y=lims, mode="lines",
                              line=dict(color=INK3, dash="dash", width=1.5),
                              hoverinfo="skip", showlegend=False))
    fig = go.Figure(data=traces)
    fig.update_layout(**BASE_LAYOUT, margin=DEFAULT_MARGIN, legend=DEFAULT_LEGEND,
                       xaxis=dict(title="Actual RUL (cycles)", gridcolor=BORDER, range=lims),
                       yaxis=dict(title="Predicted RUL (cycles)", gridcolor=BORDER, range=lims))
    return fig


def fig_rul_trajectories() -> go.Figure:
    df = pd.read_parquet(PROC_DIR / "features_DS02.parquet")
    df = df.sort_values(["unit", "cycle"])
    traces = []
    for i, unit in enumerate(sorted(df["unit"].unique())):
        sub = df[df["unit"] == unit]
        traces.append(go.Scatter(
            x=sub["cycle"], y=sub["RUL"], mode="lines", name=f"Engine {int(unit)}",
            line=dict(color=CATEGORICAL[i % len(CATEGORICAL)], width=2),
            hovertemplate="Cycle %{x}<br>RUL: %{y}<extra>Engine " + str(int(unit)) + "</extra>",
        ))
    fig = go.Figure(data=traces)
    fig.update_layout(**BASE_LAYOUT, margin=DEFAULT_MARGIN, legend=DEFAULT_LEGEND,
                       xaxis=dict(title="Flight Cycle", gridcolor=BORDER),
                       yaxis=dict(title="RUL (cycles)", gridcolor=BORDER, rangemode="tozero"))
    return fig


def fig_sdr_aircraft() -> go.Figure:
    df = pd.read_csv(RESULTS_DIR / "sdr_events_by_aircraft.csv").head(15)
    df = df.sort_values("events")
    fig = go.Figure(go.Bar(
        x=df["events"], y=df["aircraft_model"], orientation="h",
        marker=dict(color=df["events"], colorscale=[[0, "#bfdbfe"], [1, BLUE_DK]]),
        hovertemplate="%{y}: %{x} SDR events<extra></extra>",
    ))
    fig.update_layout(**BASE_LAYOUT,
                       margin=dict(l=90, r=20, t=10, b=45),
                       xaxis=dict(title="Engine SDR Events (2020-2024)", gridcolor=BORDER),
                       yaxis=dict(gridcolor=BORDER))
    return fig


def fig_sdr_trend() -> go.Figure:
    df = pd.read_csv(RESULTS_DIR / "sdr_annual_trend.csv")
    fig = go.Figure(go.Scatter(
        x=df["year"], y=df["events"], mode="lines+markers",
        line=dict(color=BLUE, width=3), marker=dict(size=9, color=BLUE),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.08)",
        hovertemplate="%{x}: %{y} events<extra></extra>",
    ))
    fig.update_layout(**BASE_LAYOUT, margin=DEFAULT_MARGIN,
                       xaxis=dict(title="Year", gridcolor=BORDER, dtick=1),
                       yaxis=dict(title="Engine SDR Events", gridcolor=BORDER, rangemode="tozero"))
    return fig


def fig_carrier_impact() -> go.Figure:
    df = pd.read_csv(RESULTS_DIR / "impact_summary.csv").nlargest(12, "avoided_cancels")
    df = df.sort_values("avoided_cancels")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["carrier_cancels"], y=df["carrier_name"], orientation="h",
        name="Total carrier cancels", marker=dict(color="#fecaca"),
        hovertemplate="%{y}: %{x:,.0f} total cancels<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=df["avoided_cancels"], y=df["carrier_name"], orientation="h",
        name="Avoidable with RUL model", marker=dict(color=BLUE),
        hovertemplate="%{y}: %{x:,.0f} avoidable<extra></extra>",
    ))
    fig.update_layout(**BASE_LAYOUT, barmode="overlay",
                       margin=dict(l=90, r=20, t=10, b=45),
                       xaxis=dict(title="Annual Cancellations", gridcolor=BORDER),
                       yaxis=dict(gridcolor=BORDER),
                       legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)))
    return fig


def fig_sensitivity() -> go.Figure:
    df = pd.read_csv(RESULTS_DIR / "sensitivity_grid.csv")
    pivot = df.pivot(index="engine_fraction", columns="warning_cycles",
                      values="system_wide_reduction_pct")
    z = pivot.values
    fig = go.Figure(go.Heatmap(
        z=z, x=[str(c) for c in pivot.columns], y=[f"{v:.0%}" for v in pivot.index],
        colorscale=[[0, "#f0fdf4"], [0.5, "#93c5fd"], [1, BLUE_DK]],
        text=[[f"{v:.1f}%" for v in row] for row in z],
        texttemplate="%{text}", textfont=dict(size=10),
        hovertemplate="Engine fraction: %{y}<br>Warning window: %{x} cycles<br>"
                      "Reduction: %{z:.1f}%<extra></extra>",
        colorbar=dict(title=dict(text="Reduction %", font=dict(size=11)),
                      thickness=12, tickfont=dict(size=10)),
    ))
    fig.update_layout(**BASE_LAYOUT,
                       margin=dict(l=60, r=20, t=10, b=45),
                       xaxis=dict(title="RUL warning window (cycles)"),
                       yaxis=dict(title="Engine fraction assumption"))
    return fig


def build_html() -> str:
    figs = {
        "rulScatter":  fig_rul_scatter(),
        "rulTraj":     fig_rul_trajectories(),
        "sdrAircraft": fig_sdr_aircraft(),
        "sdrTrend":    fig_sdr_trend(),
        "carrierImp":  fig_carrier_impact(),
        "sensGrid":    fig_sensitivity(),
    }
    fig_json = {k: pio.to_json(v) for k, v in figs.items()}

    plot_calls = "\n".join(
        f'Plotly.newPlot("{k}", {fig_json[k]}.data, {fig_json[k]}.layout, PLOTCFG);'
        for k in figs
    )

    impact = pd.read_csv(RESULTS_DIR / "impact_summary.csv")
    total_avoided  = int(impact["avoided_cancels"].sum())
    total_cancels  = int(impact["carrier_cancels"].sum())
    reduction_pct  = 100 * total_avoided / total_cancels
    sdr_events     = int(pd.read_csv(RESULTS_DIR / "sdr_annual_trend.csv")["events"].sum())
    rmse           = pd.read_csv(RESULTS_DIR / "rul_cv_results_DS02.csv")["rmse"].mean()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Engine Predictive Maintenance &mdash; Results Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@600;700&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js" charset="utf-8"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #f1f5f9; color: {INK}; line-height: 1.5;
  }}
  a {{ color: {BLUE}; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}

  header {{
    background: #ffffff; border-bottom: 1px solid {BORDER};
    padding: 1.5rem 2rem;
  }}
  .hdr-inner {{ max-width: 1320px; margin: 0 auto; display: flex; align-items: baseline;
                justify-content: space-between; flex-wrap: wrap; gap: .75rem; }}
  header h1 {{
    font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.4rem;
    letter-spacing: -0.01em; color: {INK};
  }}
  header .back {{ font-size: .85rem; font-weight: 600; color: {INK2}; }}
  .badge {{
    display:inline-block; background:#eff6ff; border:1px solid rgba(37,99,235,.18);
    color:{BLUE}; font-size:.72rem; font-weight:700; letter-spacing:.04em; text-transform:uppercase;
    padding:.2rem .6rem; border-radius:100px; margin-left:.6rem; vertical-align:middle;
  }}

  main {{ max-width: 1320px; margin: 0 auto; padding: 2rem; }}

  .kpi-row {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 1rem; margin-bottom: 2rem; }}
  .kpi {{
    background: #ffffff; border: 1px solid {BORDER}; border-radius: 12px;
    padding: 1.1rem 1.3rem; box-shadow: 0 1px 3px rgba(15,23,42,.06);
  }}
  .kpi .n {{ font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.6rem; color: {BLUE}; }}
  .kpi .l {{ font-size: .78rem; color: {INK2}; margin-top: .25rem; line-height: 1.4; }}

  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; }}
  .panel {{
    background: #ffffff; border: 1px solid {BORDER}; border-radius: 12px;
    padding: 1.4rem 1.5rem 1rem; box-shadow: 0 1px 3px rgba(15,23,42,.06);
  }}
  .panel h2 {{ font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1rem; color: {INK}; margin-bottom:.15rem; }}
  .panel .sub {{ font-size: .8rem; color: {INK3}; margin-bottom: .9rem; }}
  .panel .chart {{ width: 100%; height: 340px; }}
  .panel.wide {{ grid-column: 1 / -1; }}
  .panel.wide .chart {{ height: 380px; }}

  footer {{
    text-align: center; padding: 2rem; font-size: .82rem; color: {INK3};
  }}

  @media (max-width: 980px) {{
    .kpi-row {{ grid-template-columns: repeat(2, 1fr); }}
    .grid {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>

<header>
  <div class="hdr-inner">
    <h1>Engine Predictive Maintenance <span class="badge">Live Results</span></h1>
    <a class="back" href="./">&larr; Back to project overview</a>
  </div>
</header>

<main>
  <div class="kpi-row">
    <div class="kpi"><div class="n">{reduction_pct:.1f}%</div><div class="l">System-wide reduction in carrier cancellations</div></div>
    <div class="kpi"><div class="n">{rmse:.2f}</div><div class="l">RUL model CV RMSE (cycles), DS02</div></div>
    <div class="kpi"><div class="n">{sdr_events:,}</div><div class="l">Real FAA engine SDR events, 2020&ndash;2024</div></div>
    <div class="kpi"><div class="n">{total_cancels:,}</div><div class="l">Annual BTS carrier cancellations (baseline)</div></div>
    <div class="kpi"><div class="n">{total_avoided:,}</div><div class="l">Projected avoidable cancellations / year</div></div>
  </div>

  <div class="grid">
    <div class="panel">
      <h2>RUL Model: Predicted vs. Actual</h2>
      <div class="sub">DS02 held-out test set &middot; dashed line = perfect prediction</div>
      <div class="chart" id="rulScatter"></div>
    </div>
    <div class="panel">
      <h2>Ground-Truth RUL Trajectories</h2>
      <div class="sub">All 9 DS02 engines (dev + test) &middot; cycles remaining to failure</div>
      <div class="chart" id="rulTraj"></div>
    </div>
    <div class="panel">
      <h2>FAA SDR: Most-Reported Aircraft Models</h2>
      <div class="sub">Engine difficulty reports (JASC 7200), 2020&ndash;2024</div>
      <div class="chart" id="sdrAircraft"></div>
    </div>
    <div class="panel">
      <h2>FAA SDR: Annual Event Trend</h2>
      <div class="sub">Total engine SDR events reported per year</div>
      <div class="chart" id="sdrTrend"></div>
    </div>
    <div class="panel wide">
      <h2>Projected Impact by Carrier</h2>
      <div class="sub">Top 12 carriers by avoidable cancellations, BTS 2025 baseline</div>
      <div class="chart" id="carrierImp"></div>
    </div>
    <div class="panel wide">
      <h2>Sensitivity: How the Estimate Moves With Its Assumptions</h2>
      <div class="sub">System-wide reduction (%) across engine-fraction and RUL warning-window assumptions &mdash; the pipeline currently uses an 18% engine fraction and a 30-cycle window</div>
      <div class="chart" id="sensGrid"></div>
    </div>
  </div>
</main>

<footer>
  Built from real pipeline output &mdash; no simulated or placeholder figures.
  &middot; <a href="https://github.com/arnav64/engine-predictive-maintenance" target="_blank">Source on GitHub</a>
</footer>

<script>
var PLOTCFG = {{ responsive: true, displaylogo: false,
                  modeBarButtonsToRemove: ["toImage","sendDataToCloud","select2d","lasso2d"] }};
{plot_calls}
</script>
</body>
</html>"""


def main() -> None:
    DOCS_DIR.mkdir(exist_ok=True)
    html = build_html()
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"Written: {OUTPUT_HTML} ({OUTPUT_HTML.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

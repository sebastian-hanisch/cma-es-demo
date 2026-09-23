"""Plotly-Abbildungen der CMA-ES-Demo: Kostenlandschaft mit Kovarianz-Ellipse und Stichprobe (wachsendes Beispiel),
Vergleichs- und Sweep-Abbildungen. Achsen sind gesperrt (fixedrange)."""

import numpy as np
import plotly.graph_objects as go

import cma_constants as C

MEAN_COLOR = "#4c78a8"
SAMPLE_COLOR = "#9ecae9"
BEST_COLOR = "#54a24b"
ELLIPSE_COLOR = "#f58518"
CMA_COLOR = "#f58518"
GA_COLOR = "#4c78a8"
REF_COLOR = "#7f7f7f"
CONTOUR_SCALE = "Blues_r"

LANDSCAPE_RES = 60


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.1), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _map_layout(fig, height=430):
    fig.update_xaxes(range=[-3, C.AREA + 3], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(range=[-3, C.AREA + 3], showgrid=False, zeroline=False, showticklabels=False)
    return _base(fig, height)


def _landscape_grid(inst, res=LANDSCAPE_RES):
    xs = np.linspace(0.0, C.AREA, res)
    gx, gy = np.meshgrid(xs, xs)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    z = inst.cost(grid).reshape(res, res)
    return xs, z


def _ellipse_xy(mean, sigma, B, D, n_points=60):
    """Punkte der 1σ-Kovarianz-Ellipse: {m + σ B D z : ||z|| = 1}."""
    theta = np.linspace(0.0, 2 * np.pi, n_points)
    unit = np.stack([np.cos(theta), np.sin(theta)], axis=0)      # (2, n_points)
    pts = mean[:, None] + sigma * (B * D) @ unit
    return pts[0], pts[1]


def build_landscape(inst, grid_xy=None, best_xy=None, title=None):
    xs, z = _landscape_grid(inst)
    fig = go.Figure()
    fig.add_trace(go.Contour(x=xs, y=xs, z=z, colorscale=CONTOUR_SCALE, showscale=False, contours=dict(coloring="fill", showlines=False), hoverinfo="skip"))
    if grid_xy is not None:
        fig.add_trace(go.Scatter(x=[grid_xy[0]], y=[grid_xy[1]], mode="markers", marker=dict(size=13, symbol="star", color="#f58518", line=dict(width=1, color="white")), name="global günstigste Lage"))
    if best_xy is not None:
        fig.add_trace(go.Scatter(x=[best_xy[0]], y=[best_xy[1]], mode="markers", marker=dict(size=11, symbol="diamond", color=BEST_COLOR, line=dict(width=1, color="white")), name="bester Fund"))
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=13), x=0.02, y=0.98))
    return _map_layout(fig)


def build_growing_example(inst, generation, grid_xy):
    """Landschaft + aktuelle Stichprobe + Mittelwert + 1σ-Kovarianz-Ellipse einer Generation (`cma_algorithm.Generation`)."""
    xs, z = _landscape_grid(inst)
    fig = go.Figure()
    fig.add_trace(go.Contour(x=xs, y=xs, z=z, colorscale=CONTOUR_SCALE, showscale=False, contours=dict(coloring="fill", showlines=False), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[grid_xy[0]], y=[grid_xy[1]], mode="markers", marker=dict(size=13, symbol="star", color="#f58518", line=dict(width=1, color="white")), name="global günstigste Lage"))
    fig.add_trace(go.Scatter(x=generation.samples[:, 0], y=generation.samples[:, 1], mode="markers",
                              marker=dict(size=6, color=SAMPLE_COLOR, line=dict(width=0.5, color=MEAN_COLOR)), name="Stichprobe (λ Nachkommen)"))
    ex, ey = _ellipse_xy(generation.mean, generation.sigma, generation.B, generation.D)
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color=ELLIPSE_COLOR, width=2), name="1σ-Kovarianz-Ellipse"))
    fig.add_trace(go.Scatter(x=[generation.mean[0]], y=[generation.mean[1]], mode="markers",
                              marker=dict(size=10, symbol="x", color=ELLIPSE_COLOR, line=dict(width=1.5, color="white")), name="Mittelwert m"))
    return _map_layout(fig)


def build_sigma_curve(sigma_history):
    xs = list(range(len(sigma_history)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=sigma_history, mode="lines", line=dict(color=MEAN_COLOR, width=2.5), showlegend=False))
    fig.update_xaxes(title_text="Generation")
    fig.update_yaxes(title_text="Schrittweite σ", type="log")
    return _base(fig, 260)


def build_best_curve(best_history, reference=None):
    xs = list(range(len(best_history)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=best_history, mode="lines", line=dict(color=MEAN_COLOR, width=2.5), name="bester Fund"))
    if reference is not None:
        fig.add_hline(y=reference, line=dict(color=REF_COLOR, dash="dot"), annotation_text="globales Optimum (Gitter)", annotation_position="bottom right")
    fig.update_xaxes(title_text="Generation")
    fig.update_yaxes(title_text="Kosten")
    return _base(fig, 260)


def build_comparison(report):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"CMA-ES<br>{report['cma_small_evals']} Auswertungen"], y=[report["cma_small"]], marker_color=CMA_COLOR, showlegend=False, width=0.4))
    fig.add_trace(go.Bar(x=[f"CMA-ES<br>{report['cma_large_evals']} Auswertungen"], y=[report["cma_large"]], marker_color=CMA_COLOR, showlegend=False, width=0.4, opacity=0.6))
    fig.add_trace(go.Bar(x=[f"GA<br>{report['ga_small_evals']} Auswertungen"], y=[report["ga_small"]], marker_color=GA_COLOR, showlegend=False, width=0.4))
    fig.add_trace(go.Bar(x=[f"GA<br>{report['ga_large_evals']} Auswertungen"], y=[report["ga_large"]], marker_color=GA_COLOR, showlegend=False, width=0.4, opacity=0.6))
    fig.update_yaxes(title_text="Trefferquote im globalen Trichter", range=[0, 1], tickformat=".0%")
    return _base(fig, 360)


def build_sigma0_experiment(rows):
    xs = [r["sigma0"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["share_global"] for r in rows], mode="lines+markers", line=dict(color=CMA_COLOR, width=2.5), showlegend=False))
    fig.update_xaxes(title_text="Anfangsschrittweite σ0")
    fig.update_yaxes(title_text="Trefferquote im globalen Trichter", range=[0, 1], tickformat=".0%")
    return _base(fig, 300)


def build_sweep(rows, param_label):
    xs = [r["value"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["share_global"] for r in rows], mode="lines+markers", line=dict(color=CMA_COLOR, width=2.5), showlegend=False))
    fig.update_xaxes(title_text=param_label)
    fig.update_yaxes(title_text="Trefferquote im globalen Trichter", range=[0, 1], tickformat=".0%")
    return _base(fig, 300)

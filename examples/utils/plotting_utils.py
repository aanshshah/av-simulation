"""
Advanced plotting utilities for AV simulation visualization.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


# Color schemes for different themes
AV_COLOR_SCHEMES = {
    'default': {
        'ego': '#2E8B57',      # Sea Green
        'other': '#4169E1',    # Royal Blue
        'collision': '#DC143C', # Crimson
        'safe': '#32CD32',     # Lime Green
        'warning': '#FF8C00',  # Dark Orange
        'danger': '#FF0000',   # Red
        'neutral': '#708090'   # Slate Gray
    },
    'professional': {
        'ego': '#1f77b4',      # Professional Blue
        'other': '#ff7f0e',    # Professional Orange
        'collision': '#d62728', # Professional Red
        'safe': '#2ca02c',     # Professional Green
        'warning': '#ff9800',  # Amber
        'danger': '#f44336',   # Material Red
        'neutral': '#9e9e9e'   # Material Grey
    },
    'colorblind': {
        'ego': '#0173B2',      # Blue
        'other': '#DE8F05',    # Orange
        'collision': '#CC78BC', # Light Purple
        'safe': '#029E73',     # Green
        'warning': '#D55E00',  # Vermillion
        'danger': '#CC78BC',   # Light Purple
        'neutral': '#56B4E9'   # Sky Blue
    }
}


class AVPlotStyle:
    """Standardized plotting style for AV simulation"""

    def __init__(self, style_name='default', theme='white'):
        self.colors = AV_COLOR_SCHEMES.get(style_name, AV_COLOR_SCHEMES['default'])
        self.theme = theme
        self.setup_matplotlib_style()

    def setup_matplotlib_style(self):
        """Configure matplotlib with AV-specific styling"""
        plt.style.use('seaborn-v0_8' if self.theme == 'white' else 'dark_background')

        # Custom parameters
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'sans-serif',
            'axes.linewidth': 1.5,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.major.size': 6,
            'ytick.major.size': 6,
            'legend.frameon': True,
            'legend.fancybox': True,
            'legend.shadow': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--'
        })

    def get_plotly_template(self):
        """Get Plotly template configuration"""
        if self.theme == 'white':
            return 'plotly_white'
        else:
            return 'plotly_dark'


class TrajectoryPlotter:
    """Specialized plotter for vehicle trajectories"""

    def __init__(self, style: AVPlotStyle):
        self.style = style

    def plot_2d_trajectory(self, df: pd.DataFrame, color_by='speed',
                          vehicle_type='ego', interactive=False):
        """Plot 2D trajectory with color coding"""

        if interactive:
            return self._plot_trajectory_plotly(df, color_by, vehicle_type)
        else:
            return self._plot_trajectory_matplotlib(df, color_by, vehicle_type)

    def _plot_trajectory_matplotlib(self, df: pd.DataFrame, color_by: str,
                                   vehicle_type: str):
        """Matplotlib trajectory plot"""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Color mapping
        colors = df[color_by] if color_by in df.columns else self.style.colors[vehicle_type]

        scatter = ax.scatter(df['position_x'], df['position_y'],
                           c=colors, cmap='viridis' if color_by in df.columns else None,
                           s=20, alpha=0.7, edgecolors='black', linewidth=0.5)

        # Add colorbar if color mapping
        if color_by in df.columns:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(f'{color_by.replace("_", " ").title()}', fontweight='bold')

        # Styling
        ax.set_xlabel('Position X (m)', fontweight='bold')
        ax.set_ylabel('Position Y (m)', fontweight='bold')
        ax.set_title(f'Vehicle Trajectory - Colored by {color_by.replace("_", " ").title()}',
                    fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        return fig, ax

    def _plot_trajectory_plotly(self, df: pd.DataFrame, color_by: str,
                               vehicle_type: str):
        """Plotly interactive trajectory plot"""
        fig = go.Figure()

        if color_by in df.columns:
            color_values = df[color_by]
            colorscale = 'Viridis'
        else:
            color_values = self.style.colors[vehicle_type]
            colorscale = None

        fig.add_trace(go.Scatter(
            x=df['position_x'],
            y=df['position_y'],
            mode='markers',
            marker=dict(
                size=8,
                color=color_values,
                colorscale=colorscale,
                colorbar=dict(title=color_by.replace('_', ' ').title()) if colorscale else None,
                line=dict(width=1, color='black')
            ),
            text=[f"{col}: {val:.2f}" for col, val in zip([color_by] * len(df), df[color_by])] if color_by in df.columns else None,
            hovertemplate="Position: (%{x:.1f}, %{y:.1f})<br>%{text}<extra></extra>",
            name=f"{vehicle_type.title()} Vehicle"
        ))

        fig.update_layout(
            title=f"Interactive Vehicle Trajectory - {color_by.replace('_', ' ').title()}",
            xaxis_title="Position X (m)",
            yaxis_title="Position Y (m)",
            template=self.style.get_plotly_template(),
            hovermode='closest'
        )

        return fig

    def plot_3d_trajectory(self, df: pd.DataFrame, z_axis='timestamp'):
        """Plot 3D trajectory with time or other variable as Z-axis"""
        fig = go.Figure(data=[go.Scatter3d(
            x=df['position_x'],
            y=df['position_y'],
            z=df[z_axis],
            mode='markers+lines',
            marker=dict(
                size=5,
                color=df['speed'] if 'speed' in df.columns else 'blue',
                colorscale='Viridis',
                colorbar=dict(title="Speed (m/s)"),
                opacity=0.8
            ),
            line=dict(color='rgba(100,100,100,0.5)', width=3),
            text=[f"Speed: {s:.1f} m/s" for s in df['speed']] if 'speed' in df.columns else None,
            hovertemplate="Position: (%{x:.1f}, %{y:.1f})<br>" +
                         f"{z_axis}: %{{z:.1f}}<br>" +
                         "%{text}<extra></extra>"
        )])

        fig.update_layout(
            title="3D Vehicle Trajectory",
            scene=dict(
                xaxis_title='Position X (m)',
                yaxis_title='Position Y (m)',
                zaxis_title=z_axis.replace('_', ' ').title()
            ),
            template=self.style.get_plotly_template()
        )

        return fig


class PerformancePlotter:
    """Specialized plotter for performance metrics"""

    def __init__(self, style: AVPlotStyle):
        self.style = style

    def plot_time_series(self, df: pd.DataFrame, metrics: List[str],
                        interactive=False):
        """Plot multiple time series metrics"""

        if interactive:
            return self._plot_time_series_plotly(df, metrics)
        else:
            return self._plot_time_series_matplotlib(df, metrics)

    def _plot_time_series_matplotlib(self, df: pd.DataFrame, metrics: List[str]):
        """Matplotlib time series plot"""
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 3*n_metrics), sharex=True)

        if n_metrics == 1:
            axes = [axes]

        colors = [self.style.colors['ego'], self.style.colors['warning'],
                 self.style.colors['danger'], self.style.colors['safe']]

        for i, metric in enumerate(metrics):
            if metric in df.columns:
                axes[i].plot(df['timestamp'], df[metric],
                           color=colors[i % len(colors)], linewidth=2, alpha=0.8)
                axes[i].set_ylabel(metric.replace('_', ' ').title(), fontweight='bold')
                axes[i].grid(True, alpha=0.3)

                # Add zero line for some metrics
                if metric in ['acceleration', 'steering_angle']:
                    axes[i].axhline(y=0, color='black', linestyle='--', alpha=0.5)

        axes[-1].set_xlabel('Time (s)', fontweight='bold')
        fig.suptitle('Performance Metrics Over Time', fontsize=16, fontweight='bold')
        plt.tight_layout()

        return fig, axes

    def _plot_time_series_plotly(self, df: pd.DataFrame, metrics: List[str]):
        """Plotly interactive time series plot"""
        fig = make_subplots(
            rows=len(metrics), cols=1,
            subplot_titles=[m.replace('_', ' ').title() for m in metrics],
            shared_xaxes=True,
            vertical_spacing=0.05
        )

        colors = [self.style.colors['ego'], self.style.colors['warning'],
                 self.style.colors['danger'], self.style.colors['safe']]

        for i, metric in enumerate(metrics):
            if metric in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df[metric],
                        name=metric.replace('_', ' ').title(),
                        line=dict(color=colors[i % len(colors)], width=2),
                        hovertemplate=f"Time: %{{x:.1f}}s<br>{metric}: %{{y:.2f}}<extra></extra>"
                    ),
                    row=i+1, col=1
                )

                # Add zero line for some metrics
                if metric in ['acceleration', 'steering_angle']:
                    fig.add_hline(y=0, line_dash="dash", line_color="gray",
                                 row=i+1, col=1)

        fig.update_layout(
            title="Interactive Performance Metrics",
            template=self.style.get_plotly_template(),
            height=200*len(metrics),
            showlegend=False
        )

        fig.update_xaxes(title_text="Time (s)", row=len(metrics), col=1)

        return fig

    def plot_performance_radar(self, metrics_dict: Dict[str, float],
                              target_dict: Optional[Dict[str, float]] = None):
        """Create radar chart for performance comparison"""
        categories = list(metrics_dict.keys())
        values = list(metrics_dict.values())

        fig = go.Figure()

        # Current performance
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Current Performance',
            line_color=self.style.colors['ego']
        ))

        # Target performance (if provided)
        if target_dict:
            target_values = [target_dict.get(cat, 0) for cat in categories]
            fig.add_trace(go.Scatterpolar(
                r=target_values,
                theta=categories,
                fill='toself',
                name='Target Performance',
                line_color=self.style.colors['safe'],
                opacity=0.6
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(max(values), max(target_dict.values()) if target_dict else 0)]
                )
            ),
            title="Performance Comparison Radar Chart",
            template=self.style.get_plotly_template(),
            showlegend=True
        )

        return fig


class SafetyPlotter:
    """Specialized plotter for safety analysis"""

    def __init__(self, style: AVPlotStyle):
        self.style = style

    def plot_risk_heatmap(self, df: pd.DataFrame, x_var='speed', y_var='acceleration'):
        """Create risk analysis heatmap"""
        fig = go.Figure(data=go.Histogram2d(
            x=df[x_var],
            y=df[y_var],
            nbinsx=20,
            nbinsy=20,
            colorscale='Reds',
            hovertemplate=f"{x_var}: %{{x:.1f}}<br>{y_var}: %{{y:.2f}}<br>Count: %{{z}}<extra></extra>"
        ))

        # Add safety boundaries
        if y_var == 'acceleration':
            fig.add_hline(y=3, line_dash="dash", line_color=self.style.colors['warning'],
                         annotation_text="Hard Acceleration")
            fig.add_hline(y=-3, line_dash="dash", line_color=self.style.colors['danger'],
                         annotation_text="Hard Braking")

        fig.update_layout(
            title=f"Risk Analysis: {x_var.title()} vs {y_var.title()}",
            xaxis_title=x_var.replace('_', ' ').title(),
            yaxis_title=y_var.replace('_', ' ').title(),
            template=self.style.get_plotly_template()
        )

        return fig

    def plot_safety_timeline(self, df: pd.DataFrame, safety_events: pd.DataFrame = None):
        """Plot safety events on timeline"""
        fig = go.Figure()

        # Base timeline
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['speed'],
            mode='lines',
            name='Speed',
            line=dict(color=self.style.colors['ego'], width=2)
        ))

        # Safety events
        if safety_events is not None and not safety_events.empty:
            fig.add_trace(go.Scatter(
                x=safety_events['timestamp'],
                y=safety_events['speed'],
                mode='markers',
                name='Safety Events',
                marker=dict(
                    size=10,
                    color=self.style.colors['danger'],
                    symbol='x'
                )
            ))

        fig.update_layout(
            title="Safety Events Timeline",
            xaxis_title="Time (s)",
            yaxis_title="Speed (m/s)",
            template=self.style.get_plotly_template()
        )

        return fig


def create_publication_figure(plot_functions: List, layout_config: Dict):
    """Create publication-ready multi-panel figure"""
    n_plots = len(plot_functions)

    # Parse layout configuration
    rows = layout_config.get('rows', 1)
    cols = layout_config.get('cols', n_plots)
    figsize = layout_config.get('figsize', (4*cols, 4*rows))

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()

    # Execute plotting functions
    for i, plot_func in enumerate(plot_functions):
        if i < len(axes):
            plot_func(axes[i])

    # Hide unused axes
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')

    # Apply publication styling
    fig.suptitle(layout_config.get('title', 'AV Simulation Analysis'),
                fontsize=16, fontweight='bold')

    plt.tight_layout()

    # Save high-resolution versions
    if layout_config.get('save', False):
        output_name = layout_config.get('output_name', 'publication_figure')
        plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_name}.pdf', bbox_inches='tight')
        print(f"📊 Saved publication figure: {output_name}")

    return fig, axes


def create_interactive_dashboard(data_dict: Dict, dashboard_config: Dict):
    """Create comprehensive interactive dashboard"""

    # Parse configuration
    title = dashboard_config.get('title', 'AV Simulation Dashboard')
    subplot_titles = dashboard_config.get('subplot_titles', [])

    n_plots = len(subplot_titles)
    rows = dashboard_config.get('rows', 2)
    cols = dashboard_config.get('cols', 3)

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subplot_titles,
        specs=dashboard_config.get('specs', None)
    )

    # Add plots based on configuration
    plot_configs = dashboard_config.get('plots', [])

    for i, plot_config in enumerate(plot_configs):
        row = (i // cols) + 1
        col = (i % cols) + 1

        plot_type = plot_config.get('type', 'scatter')
        data_key = plot_config.get('data_key', 'vehicles')

        if data_key in data_dict:
            df = data_dict[data_key]

            if plot_type == 'scatter':
                fig.add_trace(
                    go.Scatter(
                        x=df[plot_config['x']],
                        y=df[plot_config['y']],
                        mode='markers',
                        name=plot_config.get('name', f'Plot {i+1}')
                    ),
                    row=row, col=col
                )
            elif plot_type == 'histogram':
                fig.add_trace(
                    go.Histogram(
                        x=df[plot_config['x']],
                        name=plot_config.get('name', f'Plot {i+1}')
                    ),
                    row=row, col=col
                )

    fig.update_layout(
        title=title,
        template='plotly_white',
        height=dashboard_config.get('height', 800)
    )

    return fig


# Convenience functions
def quick_trajectory_plot(df: pd.DataFrame, style='default', interactive=True):
    """Quick trajectory plot with sensible defaults"""
    plotter = TrajectoryPlotter(AVPlotStyle(style))
    return plotter.plot_2d_trajectory(df, interactive=interactive)


def quick_performance_plot(df: pd.DataFrame, style='default'):
    """Quick performance metrics plot"""
    plotter = PerformancePlotter(AVPlotStyle(style))
    metrics = ['speed', 'acceleration', 'steering_angle']
    available_metrics = [m for m in metrics if m in df.columns]
    return plotter.plot_time_series(df, available_metrics, interactive=True)


def quick_safety_analysis(df: pd.DataFrame, style='default'):
    """Quick safety analysis plot"""
    plotter = SafetyPlotter(AVPlotStyle(style))
    return plotter.plot_risk_heatmap(df)


if __name__ == "__main__":
    print("🎨 AV Plotting Utilities Loaded")
    print("Available functions:")
    print("  - TrajectoryPlotter: Vehicle path visualization")
    print("  - PerformancePlotter: Metrics and time series")
    print("  - SafetyPlotter: Risk analysis and safety events")
    print("  - create_publication_figure: Multi-panel publication plots")
    print("  - create_interactive_dashboard: Comprehensive dashboards")
---
layout: default
title: "Jupyter Notebooks"
---

# Interactive Jupyter Notebooks

## 📚 Available Notebooks

Our comprehensive collection of Jupyter notebooks provides hands-on tutorials and analysis tools for autonomous vehicle simulation.

### 🚀 Getting Started

#### [01_colab_setup.ipynb](examples/notebooks/01_colab_setup.ipynb)
**Google Colab Environment Setup**

- Virtual display configuration for pygame
- System dependency installation
- Environment testing and validation
- Quick setup for cloud-based development

```python
# Quick start in Colab
from examples.utils.colab_helpers import quick_colab_setup
runner = quick_colab_setup()
```

**Key Features:**
- One-click Colab environment setup
- Automatic dependency installation
- Virtual display management
- Compatibility testing

---

#### [02_simulation_runner.ipynb](examples/notebooks/02_simulation_runner.ipynb)
**Running Simulations and Data Collection**

- Headless simulation execution
- Advanced configuration options
- Batch simulation capabilities
- Screenshot capture and export

```python
# Run headless simulation
config = {
    'environment': 'highway',
    'duration': 60,
    'collect_data': True,
    'screenshot_interval': 2.0
}

result = runner.run_headless_simulation(config)
runner.display_screenshots()
```

**What You'll Learn:**
- Setting up simulation configurations
- Running multiple simulation scenarios
- Capturing and managing simulation data
- Exporting results for analysis

---

### 📊 Data Analysis

#### [03_data_analysis.ipynb](examples/notebooks/03_data_analysis.ipynb)
**Comprehensive Data Analysis Pipeline**

- Safety metrics calculation
- Trajectory analysis and visualization
- Behavioral pattern identification
- Performance optimization insights

```python
# Analyze simulation data
analyzer = SimulationAnalyzer(data_repository)

# Safety metrics
safety_metrics = analyzer.calculate_safety_metrics()
print(f"Collision rate: {safety_metrics['collision_rate']:.3f}")
print(f"Time to collision: {safety_metrics['avg_ttc']:.2f}s")

# Behavioral analysis
clusters = analyzer.analyze_driving_behaviors()
analyzer.plot_behavior_clusters(clusters)
```

**Analysis Includes:**
- **Safety Metrics**: Collision rates, time-to-collision, near-miss events
- **Trajectory Analysis**: Path smoothness, lane adherence, turning behavior
- **Performance Metrics**: Speed efficiency, fuel consumption estimates
- **Behavioral Clustering**: Aggressive vs. conservative driving patterns

---

#### [04_visualization_examples.ipynb](examples/notebooks/04_visualization_examples.ipynb)
**Advanced Visualization Techniques**

- Static publication-ready plots
- Interactive Plotly dashboards
- 3D trajectory visualization
- Animated simulation playback

```python
# Create interactive dashboard
dashboard = InteractiveDashboard(simulation_data)
dashboard.create_trajectory_plot(vehicle_id='ego')
dashboard.create_speed_heatmap()
dashboard.show()
```

**Visualization Types:**
- **Trajectory Plots**: Vehicle paths with speed coloring
- **Heatmaps**: Traffic density and collision hotspots
- **Time Series**: Speed, acceleration, and position over time
- **3D Visualizations**: Multi-dimensional trajectory analysis
- **Animated Playback**: Step-by-step simulation replay

---

#### [05_advanced_analysis.ipynb](examples/notebooks/05_advanced_analysis.ipynb)
**Statistical Modeling and Machine Learning**

- Hypothesis testing and validation
- Time series analysis and forecasting
- Machine learning for behavior prediction
- Anomaly detection algorithms

```python
# Time series analysis
ts_analyzer = TimeSeriesAnalyzer(speed_data)
forecast = ts_analyzer.arima_forecast(steps=100)
ts_analyzer.plot_forecast_with_confidence()

# Anomaly detection
anomaly_detector = AnomalyDetector()
anomalies = anomaly_detector.detect_anomalies(trajectory_data)
print(f"Found {len(anomalies)} anomalous events")
```

**Advanced Techniques:**
- **Statistical Testing**: Normality tests, stationarity analysis
- **ARIMA Modeling**: Time series forecasting
- **Machine Learning**: SVM, Random Forest for behavior prediction
- **Anomaly Detection**: Isolation Forest, statistical outliers
- **Pareto Analysis**: Multi-objective optimization

---

## 🛠️ Interactive Features

### Real-time Analysis Tools

```python
# Real-time simulation monitoring
from examples.utils.plotting_utils import RealTimePlotter

plotter = RealTimePlotter()

# During simulation
while simulation.running:
    simulation.step()

    # Update real-time plots
    plotter.update_speed_plot(ego_vehicle.speed)
    plotter.update_trajectory_plot(ego_vehicle.position)
    plotter.refresh_display()
```

### Custom Analysis Widgets

```python
# Interactive parameter exploration
import ipywidgets as widgets

@widgets.interact(
    traffic_density=(0.1, 0.8, 0.1),
    speed_limit=(20, 50, 5),
    planning_horizon=(1, 5, 0.5)
)
def explore_parameters(traffic_density, speed_limit, planning_horizon):
    # Run simulation with parameters
    result = run_simulation_with_params(
        density=traffic_density,
        speed=speed_limit,
        horizon=planning_horizon
    )

    # Display results
    plot_simulation_results(result)
```

## 📋 Notebook Structure

Each notebook follows a consistent structure:

### 1. Setup and Imports
```python
# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# AV Simulation imports
from av_simulation.core.simulation import Simulation
from av_simulation.data.repository import DataRepository
from examples.utils.plotting_utils import AVPlotStyle
```

### 2. Configuration
```python
# Set plotting style
plt.style.use('seaborn-v0_8')
AVPlotStyle.setup_matplotlib()

# Configure analysis parameters
ANALYSIS_CONFIG = {
    'simulation_duration': 60,
    'environments': ['highway', 'merge', 'roundabout'],
    'metrics': ['safety', 'efficiency', 'comfort']
}
```

### 3. Data Loading/Generation
```python
# Load existing data or run new simulation
if use_existing_data:
    data = load_simulation_data('data/simulation_results.json')
else:
    data = run_simulation_batch(ANALYSIS_CONFIG)
```

### 4. Analysis and Visualization
```python
# Perform analysis
results = analyze_simulation_data(data)

# Create visualizations
create_analysis_plots(results)
display_summary_statistics(results)
```

### 5. Conclusions and Export
```python
# Save results
export_analysis_results(results, 'analysis_output/')

# Display key findings
print_key_insights(results)
```

## 🎯 Learning Objectives

### Beginner Level (Notebooks 01-02)
- Understanding AV simulation concepts
- Setting up development environment
- Running basic simulations
- Collecting and managing data

### Intermediate Level (Notebook 03-04)
- Analyzing simulation results
- Creating effective visualizations
- Identifying patterns in driving behavior
- Optimizing simulation parameters

### Advanced Level (Notebook 05)
- Statistical hypothesis testing
- Machine learning applications
- Anomaly detection
- Performance optimization

## 💡 Usage Tips

### Running Locally
```bash
# Install Jupyter
pip install jupyter notebook

# Start notebook server
jupyter notebook

# Navigate to examples/notebooks/
```

### Running in Google Colab
1. Upload notebooks to Google Drive
2. Open with Google Colaboratory
3. Run the setup cells first
4. Follow notebook instructions

### Best Practices
- Run cells sequentially for best results
- Modify parameters to explore different scenarios
- Save your work frequently
- Export results for later analysis

## 🔧 Customization

### Adding Custom Analysis
```python
# Create custom analysis function
def custom_safety_analysis(data):
    """Your custom safety analysis"""
    # Implement your analysis logic
    results = {}

    # Calculate custom metrics
    results['custom_metric'] = calculate_custom_metric(data)

    return results

# Use in notebook
custom_results = custom_safety_analysis(simulation_data)
```

### Custom Visualizations
```python
# Create custom plotting function
def plot_custom_metric(data, metric_name):
    """Plot custom metric over time"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Your plotting logic here
    ax.plot(data.timestamps, data[metric_name])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(metric_name)

    plt.show()
    return fig
```

## 📖 Additional Resources

### Documentation Links
- [API Reference](api.html) - Detailed function documentation
- [Examples](examples.html) - Code examples and tutorials
- [Documentation](documentation.html) - Complete user guide

### External Resources
- [Jupyter Documentation](https://jupyter.org/documentation)
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)
- [Plotly Python Guide](https://plotly.com/python/)

### Community
- [GitHub Discussions](https://github.com/aanshshah/av-simulation/discussions)
- [Issue Tracker](https://github.com/aanshshah/av-simulation/issues)

---

**Ready to start?**
1. Begin with [01_colab_setup.ipynb](examples/notebooks/01_colab_setup.ipynb) for environment setup
2. Continue with [02_simulation_runner.ipynb](examples/notebooks/02_simulation_runner.ipynb) to run your first simulation
3. Explore the analysis notebooks based on your interests

**Need help?** Check our [Documentation](documentation.html) or open an issue on [GitHub](https://github.com/aanshshah/av-simulation/issues).
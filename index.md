---
layout: default
title: "Autonomous Vehicle Simulation Environment"
---

# Autonomous Vehicle Simulation Environment

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/av-simulation)](https://pypi.org/project/av-simulation/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://aanshshah.github.io/av-simulation)

> Recreation of the simulation environment from the paper: **"Safety and Risk Analysis of Autonomous Vehicles Using Computer Vision and Neural Networks"** by Dixit et al. (2021)

## 🚗 Overview

This project recreates the three main case studies from the research paper, providing a comprehensive autonomous vehicle simulation environment with:

- **Lane Detection Algorithms** (straight and curved lanes)
- **Behavioral Planning** with reinforcement learning
- **Risk Analysis** and safety metrics
- **Data Collection** and analysis tools
- **Interactive Jupyter Notebooks** for research and education

## ✨ Key Features

### 🛣️ Three Simulation Environments
- **Highway Environment**: 4-lane highway with multiple vehicles
- **Lane Merging**: Highway with service road merge scenarios
- **Roundabout**: Complex 4-way roundabout navigation

### 👁️ Computer Vision
- **Straight Lane Detection** using Hough Line Transform
- **Curved Lane Detection** with HSV color space and polynomial fitting
- Real-time lane boundary detection and tracking

### 🧠 AI & Machine Learning
- **Model-based Reinforcement Learning** for behavioral planning
- **Neural Network Dynamics** modeling
- **Robust Control** with uncertainty handling
- **Collision Prediction** and avoidance

### 📊 Data Analysis & Visualization
- **Comprehensive Data Repository** for simulation metrics
- **Real-time Data Collection** during simulation runs
- **Advanced Analytics** with statistical modeling
- **Interactive Visualizations** using Plotly and Matplotlib

## 🚀 Quick Start

### Installation

**Option 1: Install from PyPI (Recommended)**
```bash
pip install av-simulation
```

**Option 2: Install from Source**
```bash
# Clone the repository
git clone https://github.com/aanshshah/av-simulation.git
cd av-simulation

# Install dependencies
pip install -r requirements.txt
```

### Run the Simulation

**From PyPI Installation:**
```bash
av-simulation
```

**From Source:**
```bash
python src/av_simulation/core/simulation.py
```

**Controls:**
- `1`, `2`, `3` - Switch between environments
- `SPACE` - Pause/Resume
- `R` - Reset current environment
- `ESC` - Exit

## 📚 Documentation

- [**Getting Started**](documentation.html) - Installation and basic usage
- [**API Reference**](api.html) - Detailed code documentation
- [**Examples**](examples.html) - Code examples and tutorials
- [**Jupyter Notebooks**](notebooks.html) - Interactive analysis and visualization

## 🔬 Research Applications

This simulation environment is designed for:

- **Autonomous Vehicle Research** - Test algorithms in controlled environments
- **Safety Analysis** - Evaluate collision avoidance and risk metrics
- **Educational Purposes** - Learn AV concepts through interactive examples
- **Data Science** - Analyze driving behaviors and performance metrics

## 📈 Example Results

### Lane Detection
![Lane Detection Example](assets/images/lane_detection_demo.png)

### Simulation Environments
![Simulation Environments](assets/images/simulation_environments.png)

### Data Analysis Dashboard
![Data Analysis](assets/images/data_analysis_dashboard.png)

## 🎯 Key Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Acceleration Range | (-5, 5.0) m/s² |
| Steering Range | ±45 degrees |
| Max Speed | 40 m/s |
| Default Speeds | [23, 25] m/s |
| Perception Distance | 180 m |

## 🔗 Quick Links

<div class="grid-container">
  <div class="grid-item">
    <h3>🚀 <a href="examples.html">Examples</a></h3>
    <p>Ready-to-run code examples and tutorials</p>
  </div>
  <div class="grid-item">
    <h3>📓 <a href="notebooks.html">Notebooks</a></h3>
    <p>Interactive Jupyter notebooks for analysis</p>
  </div>
  <div class="grid-item">
    <h3>📖 <a href="documentation.html">Documentation</a></h3>
    <p>Comprehensive API and usage documentation</p>
  </div>
  <div class="grid-item">
    <h3>🔧 <a href="api.html">API Reference</a></h3>
    <p>Detailed function and class references</p>
  </div>
</div>

## 📄 Citation

If you use this simulation environment in your research, please cite:

```bibtex
@article{dixit2021safety,
  title={Safety and Risk Analysis of Autonomous Vehicles Using Computer Vision and Neural Networks},
  author={Dixit, A. and Kumar Chidambaram, R. and Allam, Z.},
  journal={Vehicles},
  volume={3},
  pages={595--617},
  year={2021}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📞 Support

- 📖 [Documentation](documentation.html)
- 🐛 [Issue Tracker](https://github.com/aanshshah/av-simulation/issues)
- 💬 [Discussions](https://github.com/aanshshah/av-simulation/discussions)

---

<div style="text-align: center; margin-top: 2rem; padding: 1rem; background-color: #f8f9fa; border-radius: 8px;">
  <p><strong>Ready to get started?</strong></p>
  <a href="documentation.html" class="btn btn-primary">View Documentation</a>
  <a href="examples.html" class="btn btn-secondary">See Examples</a>
</div>
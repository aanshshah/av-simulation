---
layout: default
title: "Documentation"
---

# Documentation

## 📋 Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Components](#core-components)
4. [Simulation Environments](#simulation-environments)
5. [Data Collection](#data-collection)
6. [Lane Detection](#lane-detection)
7. [Behavioral Planning](#behavioral-planning)
8. [Configuration](#configuration)
9. [Troubleshooting](#troubleshooting)

## 🚀 Installation

### Prerequisites

- Python 3.10 or 3.11 (Miniforge/conda-forge builds recommended on Apple Silicon)
- git (if you plan to clone the repository)
- An environment manager (`python -m venv`, `uv`, or `conda`)

### Option 1: Reproducible Local Setup

```bash
# Clone the repository
git clone https://github.com/aanshshah/av-simulation.git
cd av-simulation

# Create and activate an isolated environment
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
# uv alternative
# uv venv && source .venv/bin/activate

# Install pinned dependencies and the package
pip install --upgrade pip
pip install -r requirements-local.txt
pip install -e .

# Optional smoke tests
python -m av_simulation.core.simulation --help
pytest tests  # requires pytest
```

`requirements-local.txt` mirrors the Colab environment but pins versions so macOS, Windows, and Linux hosts resolve the same wheels (pygame, OpenCV, torch, etc.).

### Option 2: Install from PyPI (Quick Evaluation)

```bash
pip install av-simulation
```

### Option 3: Google Colab

```python
# Install in Colab
!pip install av-simulation pyvirtualdisplay

# Or use the bundled helpers
from examples.utils.colab_helpers import quick_colab_setup
runner = quick_colab_setup()
```

## ⚡ Quick Start

### Basic Simulation

```python
from av_simulation.core.simulation import Simulation

# Create and run simulation
sim = Simulation()
sim.run()
```

### With Data Collection

```python
from av_simulation.core.simulation import Simulation

# Enable data collection
sim = Simulation(enable_data_collection=True)
sim.run()

# Export collected data
sim.data_repository.export_to_csv("simulation_data.csv")
```

### Command Line Usage

**From PyPI Installation:**
```bash
# Run main simulation
av-simulation
```

**From Source Installation:**
```bash
# Run main simulation
python src/av_simulation/core/simulation.py

# Run lane detection demo
python src/av_simulation/detection/lane_detection.py

# Run behavioral planning demo
python src/av_simulation/planning/behavioral_planning.py
```

## 🔧 Core Components

### Simulation Engine

The main simulation engine coordinates all components:

```python
class Simulation:
    def __init__(self, enable_data_collection=False):
        # Initialize pygame and create screen
        # Setup environments (highway, merge, roundabout)
        # Initialize ego vehicle and traffic
        # Setup data collection if enabled
```

**Key Methods:**
- `run()` - Start the main simulation loop
- `switch_environment(env_type)` - Change simulation environment
- `collect_simulation_data()` - Collect data for current state

### Vehicle Models

#### Ego Vehicle
The autonomous vehicle with AI planning:

```python
class EgoVehicle:
    def __init__(self, x, y, heading=0):
        self.state = VehicleState(x, y, vx=25, vy=0, heading=heading)
        self.planner = BehavioralPlanner()
        self.controller = VehicleController()
```

#### Traffic Vehicles
Simulated traffic with realistic behaviors:

```python
class TrafficVehicle:
    def __init__(self, x, y, lane, speed):
        self.behavior = random.choice(['aggressive', 'normal', 'conservative'])
        self.target_speed = speed
```

### Data Repository

Thread-safe data collection and storage:

```python
from av_simulation.data.repository import DataRepository

repo = DataRepository()
repo.start_new_run(config={'environment': 'highway'})
repo.add_vehicle_snapshot(vehicle_data)
repo.export_to_csv("output.csv")
```

## 🛣️ Simulation Environments

### 1. Highway Environment

4-lane highway with steady traffic flow:

```python
sim.switch_environment('highway')
```

**Features:**
- Multiple lanes with lane changing
- Various traffic densities
- Speed limit enforcement
- Collision detection

### 2. Merge Environment

Highway with service road merging:

```python
sim.switch_environment('merge')
```

**Features:**
- Complex merging scenarios
- Yield behavior modeling
- Gap acceptance algorithms
- Merge conflict resolution

### 3. Roundabout Environment

4-way roundabout navigation:

```python
sim.switch_environment('roundabout')
```

**Features:**
- Circular traffic flow
- Yield-to-left rules
- Entry/exit gap detection
- Multi-lane roundabout support

## 📊 Data Collection

### Enabling Data Collection

```python
# Enable during initialization
sim = Simulation(enable_data_collection=True)

# Or enable later
sim.enable_data_collection()
```

### Data Types Collected

#### Vehicle Snapshots
```python
@dataclass
class VehicleSnapshot:
    vehicle_id: str
    timestamp: float
    position_x: float
    position_y: float
    velocity_x: float
    velocity_y: float
    heading: float
    acceleration: float
    lane_id: int
```

#### Environment Data
```python
@dataclass
class EnvironmentSnapshot:
    timestamp: float
    environment_type: str
    traffic_density: float
    average_speed: float
    collision_count: int
```

### Export Options

```python
# CSV export
repo.export_to_csv("data.csv")

# JSON export
repo.export_to_json("data.json")

# HDF5 export (for large datasets)
repo.export_to_hdf5("data.h5")
```

## 👁️ Lane Detection

### Straight Lane Detection

Based on Hough Line Transform:

```python
from av_simulation.perception.lane_detection import StraightLaneDetector

detector = StraightLaneDetector()
lanes = detector.detect_lanes(image)
```

**Algorithm Steps:**
1. Grayscale conversion
2. Gaussian blur (5×5 kernel)
3. Canny edge detection
4. Region of interest masking
5. Hough line detection
6. Line filtering and averaging

### Curved Lane Detection

Using polynomial fitting:

```python
from av_simulation.perception.lane_detection import CurvedLaneDetector

detector = CurvedLaneDetector()
left_fit, right_fit = detector.detect_curved_lanes(image)
```

**Algorithm Steps:**
1. Camera calibration correction
2. Perspective transformation
3. HSV color space filtering
4. Binary thresholding
5. Sliding window search
6. Polynomial fitting (2nd degree)
7. Curvature calculation

## 🧠 Behavioral Planning

### MDP-Based Planning

The behavioral planner uses Markov Decision Processes:

```python
from av_simulation.planning.behavioral_planning import BehavioralPlanner

planner = BehavioralPlanner()
action = planner.plan_action(ego_vehicle, environment)
```

### Action Space

Available actions for the ego vehicle:

```python
class Action:
    MAINTAIN_SPEED = 0
    ACCELERATE = 1
    DECELERATE = 2
    CHANGE_LANE_LEFT = 3
    CHANGE_LANE_RIGHT = 4
```

### State Representation

Vehicle and environment state:

```python
@dataclass
class PlanningState:
    ego_position: Tuple[float, float]
    ego_velocity: Tuple[float, float]
    front_vehicle_distance: float
    left_lane_clear: bool
    right_lane_clear: bool
    target_speed: float
```

### Reward Function

The planner optimizes for:
- **Safety**: Collision avoidance (-1000 penalty)
- **Efficiency**: Speed maintenance (+10 reward)
- **Comfort**: Smooth acceleration (+5 reward)
- **Traffic Flow**: Proper lane usage (+3 reward)

## ⚙️ Configuration

### Simulation Parameters

Edit parameters in `config.py`:

```python
# Vehicle dynamics
MAX_ACCELERATION = 5.0  # m/s²
MAX_DECELERATION = -5.0  # m/s²
MAX_STEERING = 0.785  # radians (45 degrees)
MAX_SPEED = 40.0  # m/s

# Perception
PERCEPTION_RANGE = 180.0  # meters
LANE_WIDTH = 3.5  # meters

# Planning
PLANNING_HORIZON = 3.0  # seconds
SAFETY_DISTANCE = 10.0  # meters
TIME_HEADWAY = 1.5  # seconds
```

### Environment Configuration

```python
# Highway environment
HIGHWAY_CONFIG = {
    'num_lanes': 4,
    'lane_width': 3.5,
    'length': 2000,
    'traffic_density': 0.3
}

# Merge environment
MERGE_CONFIG = {
    'main_road_lanes': 3,
    'merge_lane_length': 200,
    'merge_angle': 15  # degrees
}
```

### Data Collection Settings

```python
DATA_CONFIG = {
    'collection_frequency': 10,  # Hz
    'buffer_size': 10000,
    'auto_export': True,
    'export_format': 'csv'
}
```

## 🐛 Troubleshooting

### Common Issues

#### "pygame not found"
```bash
pip install pygame
```

#### "No display found" (Linux)
```bash
# Install virtual display
sudo apt-get install xvfb
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x24 &
```

#### "Module not found" errors
```bash
# Ensure you're in the project directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

#### Performance Issues
- Reduce visualization frequency
- Disable data collection for faster runs
- Lower traffic density in environments

### Debug Mode

Enable debug output:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

sim = Simulation(debug=True)
```

### Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_simulation.py

# Run with coverage
python -m pytest tests/ --cov=av_simulation
```

## 📚 Additional Resources

- [API Reference](api.html) - Detailed class and function documentation
- [Examples](examples.html) - Code examples and tutorials
- [Jupyter Notebooks](notebooks.html) - Interactive analysis guides
- [GitHub Repository](https://github.com/aanshshah/av-simulation) - Source code and issues

## 💡 Tips for Best Results

1. **Start Simple**: Begin with the highway environment
2. **Enable Data Collection**: Use data to understand vehicle behaviors
3. **Experiment with Parameters**: Adjust planning and control parameters
4. **Use Notebooks**: Interactive analysis provides better insights
5. **Monitor Performance**: Watch collision rates and efficiency metrics

---

**Need help?** Open an issue on [GitHub](https://github.com/aanshshah/av-simulation/issues) or check our [FAQ](faq.html).
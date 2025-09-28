---
layout: default
title: "API Reference"
---

# API Reference

## 📋 Table of Contents

1. [Core Simulation](#core-simulation)
2. [Vehicle Models](#vehicle-models)
3. [Environment System](#environment-system)
4. [Data Collection](#data-collection)
5. [Perception System](#perception-system)
6. [Planning System](#planning-system)
7. [Utilities](#utilities)

---

## 🎮 Core Simulation

### `av_simulation.core.simulation.Simulation`

Main simulation engine that coordinates all components.

```python
class Simulation:
    def __init__(self, width=1200, height=800, enable_data_collection=False,
                 headless=False, data_repository=None, debug=False)
```

**Parameters:**
- `width` (int): Screen width in pixels. Default: 1200
- `height` (int): Screen height in pixels. Default: 800
- `enable_data_collection` (bool): Enable data collection. Default: False
- `headless` (bool): Run without GUI. Default: False
- `data_repository` (DataRepository): Custom data repository. Default: None
- `debug` (bool): Enable debug output. Default: False

#### Methods

##### `run()`
Start the main simulation loop with GUI.

```python
sim = Simulation()
sim.run()
```

##### `step()`
Execute a single simulation step.

```python
sim = Simulation(headless=True)
while condition:
    sim.step()
```

##### `switch_environment(env_type: str)`
Switch to a different environment.

```python
sim.switch_environment('highway')  # 'highway', 'merge', 'roundabout'
```

##### `reset()`
Reset the current environment to initial state.

```python
sim.reset()
```

##### `get_performance_metrics() -> Dict`
Get current performance metrics.

```python
metrics = sim.get_performance_metrics()
# Returns: {'collision_rate': 0.05, 'avg_speed': 24.5, 'efficiency': 0.87}
```

---

## 🚗 Vehicle Models

### `av_simulation.vehicles.ego_vehicle.EgoVehicle`

The autonomous ego vehicle with AI planning capabilities.

```python
class EgoVehicle:
    def __init__(self, x: float, y: float, heading: float = 0,
                 lane: int = 1, planner=None)
```

**Parameters:**
- `x` (float): Initial x position
- `y` (float): Initial y position
- `heading` (float): Initial heading in radians
- `lane` (int): Initial lane number
- `planner` (BehavioralPlanner): Custom planner instance

#### Properties

##### `state: VehicleState`
Current vehicle state containing position, velocity, and heading.

```python
print(f"Position: ({vehicle.state.x}, {vehicle.state.y})")
print(f"Speed: {vehicle.state.vx} m/s")
print(f"Heading: {vehicle.state.heading} rad")
```

##### `action: Action`
Current action being executed.

```python
if vehicle.action == Action.ACCELERATE:
    print("Vehicle is accelerating")
```

#### Methods

##### `set_action(action: Action)`
Set the action for the vehicle to execute.

```python
vehicle.set_action(Action.CHANGE_LANE_LEFT)
```

##### `step(dt: float, environment)`
Update vehicle state for one time step.

```python
vehicle.step(1.0/60, environment)  # 60 FPS
```

##### `get_sensor_data(environment) -> Dict`
Get sensor readings from the environment.

```python
sensors = vehicle.get_sensor_data(environment)
# Returns: {'front_distance': 45.2, 'left_clear': True, 'right_clear': False}
```

### `av_simulation.vehicles.traffic_vehicle.TrafficVehicle`

Non-autonomous traffic vehicles with predefined behaviors.

```python
class TrafficVehicle:
    def __init__(self, x: float, y: float, lane: int, speed: float,
                 behavior: str = 'normal')
```

**Parameters:**
- `x` (float): Initial x position
- `y` (float): Initial y position
- `lane` (int): Lane assignment
- `speed` (float): Target speed in m/s
- `behavior` (str): Driving behavior ('aggressive', 'normal', 'conservative')

#### Methods

##### `update_behavior(new_behavior: str)`
Change the vehicle's driving behavior.

```python
traffic_vehicle.update_behavior('aggressive')
```

---

## 🌍 Environment System

### `av_simulation.environments.base_environment.BaseEnvironment`

Base class for all simulation environments.

```python
class BaseEnvironment:
    def __init__(self, width: int, height: int)
```

#### Methods

##### `step(dt: float)`
Update environment for one time step.

##### `draw(screen)`
Render the environment to the screen.

##### `configure(config: Dict)`
Apply configuration parameters.

```python
env.configure({
    'traffic_density': 0.4,
    'speed_limit': 35,
    'weather': 'clear'
})
```

### `av_simulation.environments.highway.HighwayEnvironment`

4-lane highway environment with steady traffic flow.

```python
class HighwayEnvironment(BaseEnvironment):
    def __init__(self, num_lanes: int = 4, lane_width: float = 3.5)
```

**Configuration Options:**
- `traffic_density` (float): 0.0-1.0, vehicle density
- `speed_limit` (float): Maximum allowed speed
- `num_lanes` (int): Number of highway lanes
- `lane_width` (float): Width of each lane in meters

### `av_simulation.environments.merge.MergeEnvironment`

Highway with service road merging scenarios.

```python
class MergeEnvironment(BaseEnvironment):
    def __init__(self, merge_length: float = 200)
```

**Configuration Options:**
- `merge_length` (float): Length of merge lane in meters
- `merge_traffic_rate` (float): Rate of merging vehicles
- `merge_angle` (float): Merge angle in degrees

### `av_simulation.environments.roundabout.RoundaboutEnvironment`

4-way roundabout with circular traffic flow.

```python
class RoundaboutEnvironment(BaseEnvironment):
    def __init__(self, radius: float = 50, num_exits: int = 4)
```

**Configuration Options:**
- `radius` (float): Roundabout radius in meters
- `num_exits` (int): Number of exit points
- `entry_rate` (float): Vehicle entry rate per second

---

## 📊 Data Collection

### `av_simulation.data.repository.DataRepository`

Thread-safe data storage and management system.

```python
class DataRepository:
    def __init__(self, max_runs: int = 100)
```

#### Methods

##### `start_new_run(config: Dict = None) -> str`
Start a new data collection run.

```python
run_id = repository.start_new_run({
    'experiment': 'highway_test',
    'traffic_density': 0.4
})
```

##### `add_vehicle_snapshot(snapshot: VehicleSnapshot)`
Add vehicle state data.

```python
snapshot = VehicleSnapshot(
    vehicle_id='ego',
    timestamp=time.time(),
    position_x=vehicle.state.x,
    position_y=vehicle.state.y,
    velocity_x=vehicle.state.vx,
    velocity_y=vehicle.state.vy,
    heading=vehicle.state.heading,
    acceleration=vehicle.acceleration,
    lane_id=vehicle.lane
)
repository.add_vehicle_snapshot(snapshot)
```

##### `export_to_csv(filename: str, run_ids: List[str] = None)`
Export data to CSV format.

```python
repository.export_to_csv("simulation_data.csv")
```

##### `export_to_json(filename: str, run_ids: List[str] = None)`
Export data to JSON format.

```python
repository.export_to_json("simulation_data.json")
```

##### `get_run_data(run_id: str) -> SimulationData`
Get data for a specific run.

```python
data = repository.get_run_data(run_id)
print(f"Collected {len(data.vehicle_snapshots)} vehicle snapshots")
```

### `av_simulation.data.collectors.DataCollectionManager`

Coordinates multiple data collectors.

```python
class DataCollectionManager:
    def __init__(self, repository: DataRepository)
```

#### Methods

##### `set_callback(callback: Callable)`
Set callback for real-time data processing.

```python
def data_callback(data_type, data):
    if data_type == 'collision':
        print(f"Collision detected: {data}")

manager.set_callback(data_callback)
```

##### `collect(simulation_state)`
Collect data from current simulation state.

```python
manager.collect(simulation.get_current_state())
```

---

## 👁️ Perception System

### `av_simulation.perception.lane_detection.StraightLaneDetector`

Detects straight lane markings using Hough Line Transform.

```python
class StraightLaneDetector:
    def __init__(self, canny_low: int = 50, canny_high: int = 150,
                 hough_threshold: int = 20)
```

#### Methods

##### `detect_lanes(image: np.ndarray) -> List[LaneLine]`
Detect lane lines in an image.

```python
detector = StraightLaneDetector()
lanes = detector.detect_lanes(road_image)

for lane in lanes:
    print(f"Lane: slope={lane.slope:.2f}, intercept={lane.intercept:.2f}")
```

##### `configure_parameters(canny_low: int, canny_high: int, hough_threshold: int)`
Update detection parameters.

```python
detector.configure_parameters(canny_low=40, canny_high=120, hough_threshold=25)
```

### `av_simulation.perception.lane_detection.CurvedLaneDetector`

Detects curved lanes using polynomial fitting.

```python
class CurvedLaneDetector:
    def __init__(self, window_width: int = 50, window_height: int = 80,
                 margin: int = 100)
```

#### Methods

##### `detect_curved_lanes(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]`
Detect curved lane boundaries.

```python
detector = CurvedLaneDetector()
left_fit, right_fit, curvature = detector.detect_curved_lanes(road_image)

print(f"Left lane polynomial: {left_fit}")
print(f"Curvature radius: {curvature:.1f} meters")
```

---

## 🧠 Planning System

### `av_simulation.planning.behavioral_planning.BehavioralPlanner`

AI-based behavioral planning for autonomous vehicles.

```python
class BehavioralPlanner:
    def __init__(self, planning_horizon: float = 3.0,
                 safety_distance: float = 10.0)
```

#### Methods

##### `plan_action(ego_vehicle, environment) -> Action`
Plan the next action for the ego vehicle.

```python
planner = BehavioralPlanner()
action = planner.plan_action(ego_vehicle, environment)

if action == Action.CHANGE_LANE_LEFT:
    print("Planning lane change to the left")
```

##### `get_current_state(ego_vehicle, environment) -> PlanningState`
Get the current planning state representation.

```python
state = planner.get_current_state(ego_vehicle, environment)
print(f"Front vehicle distance: {state.front_vehicle_distance:.1f}m")
```

##### `update_parameters(safety_distance: float, planning_horizon: float)`
Update planning parameters.

```python
planner.update_parameters(safety_distance=15.0, planning_horizon=4.0)
```

### `av_simulation.planning.actions.Action`

Enumeration of available vehicle actions.

```python
class Action(Enum):
    MAINTAIN_SPEED = 0
    ACCELERATE = 1
    DECELERATE = 2
    CHANGE_LANE_LEFT = 3
    CHANGE_LANE_RIGHT = 4
    EMERGENCY_BRAKE = 5
```

---

## 🛠️ Utilities

### `examples.utils.plotting_utils.AVPlotStyle`

Consistent styling for autonomous vehicle plots.

```python
class AVPlotStyle:
    @staticmethod
    def setup_matplotlib()
```

#### Usage

```python
from examples.utils.plotting_utils import AVPlotStyle

AVPlotStyle.setup_matplotlib()
# Now all plots use consistent AV simulation styling
```

### `examples.utils.colab_helpers.ColabDisplayManager`

Google Colab virtual display management.

```python
class ColabDisplayManager:
    def setup_virtual_display(self, width: int = 1200, height: int = 800) -> bool
```

#### Methods

##### `setup_virtual_display(width, height) -> bool`
Setup virtual display for Colab.

```python
display_manager = ColabDisplayManager()
success = display_manager.setup_virtual_display(1200, 800)
```

##### `test_pygame() -> bool`
Test pygame functionality.

```python
if display_manager.test_pygame():
    print("Pygame working correctly")
```

---

## 📚 Data Structures

### `VehicleState`

```python
@dataclass
class VehicleState:
    x: float          # Position x (meters)
    y: float          # Position y (meters)
    vx: float         # Velocity x (m/s)
    vy: float         # Velocity y (m/s)
    heading: float    # Heading angle (radians)
    lane: int         # Current lane number
```

### `VehicleSnapshot`

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
    action_type: str
```

### `CollisionEvent`

```python
@dataclass
class CollisionEvent:
    timestamp: float
    vehicle1_id: str
    vehicle2_id: str
    position_x: float
    position_y: float
    severity: float
    collision_type: str
```

---

## 🔧 Configuration

### Global Configuration

Access simulation-wide settings:

```python
from av_simulation.config import SimulationConfig

config = SimulationConfig()
config.MAX_SPEED = 50.0  # m/s
config.PERCEPTION_RANGE = 200.0  # meters
config.PLANNING_FREQUENCY = 10  # Hz
```

### Environment-Specific Configuration

```python
# Highway configuration
highway_config = {
    'num_lanes': 4,
    'lane_width': 3.5,
    'speed_limit': 35.0,
    'traffic_density': 0.3
}

# Apply configuration
environment.configure(highway_config)
```

---

## 🐛 Error Handling

### Common Exceptions

#### `SimulationError`
General simulation-related errors.

```python
try:
    sim.run()
except SimulationError as e:
    print(f"Simulation error: {e}")
```

#### `DataCollectionError`
Data collection and storage errors.

```python
try:
    repository.export_to_csv("data.csv")
except DataCollectionError as e:
    print(f"Data export failed: {e}")
```

#### `PlanningError`
Planning algorithm errors.

```python
try:
    action = planner.plan_action(vehicle, environment)
except PlanningError as e:
    print(f"Planning failed: {e}")
    action = Action.EMERGENCY_BRAKE
```

---

## 📈 Performance Considerations

### Memory Management

- Data repositories automatically manage memory with configurable limits
- Use `repository.clear_old_runs()` to free memory
- Consider exporting data periodically for long simulations

### CPU Optimization

- Use `headless=True` for faster simulation without graphics
- Adjust `PLANNING_FREQUENCY` to balance accuracy vs. performance
- Consider parallel processing for batch simulations

### Data Export Optimization

- Use HDF5 format for large datasets
- Export in batches for continuous data collection
- Compress exported files to save storage space

---

For more detailed examples and usage patterns, see:
- [Examples](examples.html) - Practical code examples
- [Notebooks](notebooks.html) - Interactive tutorials
- [Documentation](documentation.html) - User guide
- [GitHub Repository](https://github.com/aanshshah/av-simulation) - Source code
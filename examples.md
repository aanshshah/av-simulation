---
layout: default
title: "Examples"
---

# Examples

## 📋 Table of Contents

1. [Basic Usage](#basic-usage)
2. [Data Collection](#data-collection)
3. [Lane Detection](#lane-detection)
4. [Behavioral Planning](#behavioral-planning)
5. [Environment Switching](#environment-switching)
6. [Custom Scenarios](#custom-scenarios)
7. [Performance Analysis](#performance-analysis)

## 🚀 Basic Usage

### Running Your First Simulation

```python
from av_simulation.core.simulation import Simulation

# Create simulation instance
sim = Simulation()

# Start the simulation
sim.run()
```

**Controls while running:**
- `1`, `2`, `3` - Switch environments
- `SPACE` - Pause/Resume
- `R` - Reset
- `ESC` - Exit

### Headless Simulation (No GUI)

```python
from av_simulation.core.simulation import Simulation
import time

# Create simulation without display
sim = Simulation(headless=True, enable_data_collection=True)

# Run for 30 seconds
start_time = time.time()
while time.time() - start_time < 30:
    sim.step()  # Single simulation step
    time.sleep(1/60)  # 60 FPS

# Get collected data
data = sim.data_repository.get_all_data()
print(f"Collected {len(data.vehicle_snapshots)} vehicle snapshots")
```

## 📊 Data Collection

### Basic Data Collection

```python
from av_simulation.core.simulation import Simulation

# Enable data collection
sim = Simulation(enable_data_collection=True)

# Run simulation
sim.run()

# Export data when done
sim.data_repository.export_to_csv("my_simulation_data.csv")
```

### Advanced Data Collection with Configuration

```python
from av_simulation.data.repository import DataRepository
from av_simulation.core.simulation import Simulation

# Create custom data repository
repo = DataRepository()

# Start a new experimental run
run_config = {
    'experiment_name': 'highway_safety_test',
    'environment': 'highway',
    'traffic_density': 0.4,
    'weather': 'clear',
    'notes': 'Testing collision avoidance with high traffic'
}

run_id = repo.start_new_run(run_config)
print(f"Started run: {run_id}")

# Create simulation with custom repository
sim = Simulation(data_repository=repo, enable_data_collection=True)
sim.switch_environment('highway')

# Run simulation
sim.run()

# End the run and export
repo.end_current_run()
repo.export_to_json(f"experiment_{run_id}.json")
```

### Real-time Data Monitoring

```python
from av_simulation.core.simulation import Simulation
from av_simulation.data.collectors import DataCollectionManager
import time

def data_callback(data_type, data):
    """Called whenever new data is collected"""
    if data_type == 'collision':
        print(f"⚠️  COLLISION detected at {data.timestamp:.2f}s")
    elif data_type == 'vehicle':
        if data.vehicle_id == 'ego':
            print(f"🚗 Ego speed: {data.speed:.1f} m/s")

# Setup simulation with callback
sim = Simulation(enable_data_collection=True)
sim.data_collection_manager.set_callback(data_callback)

# Run with real-time monitoring
sim.run()
```

## 👁️ Lane Detection

### Straight Lane Detection Example

```python
from av_simulation.perception.lane_detection import StraightLaneDetector
import cv2
import numpy as np

# Create detector
detector = StraightLaneDetector()

# Create a test road image
def create_test_road():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw road surface
    cv2.rectangle(img, (0, 300), (640, 480), (50, 50, 50), -1)
    # Draw lane markings
    cv2.line(img, (200, 480), (250, 300), (255, 255, 255), 3)
    cv2.line(img, (390, 480), (440, 300), (255, 255, 255), 3)
    return img

# Test detection
test_image = create_test_road()
lanes = detector.detect_lanes(test_image)

print(f"Detected {len(lanes)} lane lines")
for i, lane in enumerate(lanes):
    print(f"Lane {i}: slope={lane.slope:.2f}, intercept={lane.intercept:.2f}")
```

### Curved Lane Detection Example

```python
from av_simulation.perception.lane_detection import CurvedLaneDetector
import numpy as np

# Create detector
detector = CurvedLaneDetector()

# Create curved road test image
def create_curved_road():
    img = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Create curved lane markings using polynomial
    y_coords = np.linspace(400, 720, 100)

    # Left lane (quadratic curve)
    left_x = 200 + 0.0003 * (y_coords - 400)**2
    # Right lane
    right_x = 1080 - 0.0003 * (y_coords - 400)**2

    # Draw curves
    for i in range(len(y_coords)-1):
        y1, y2 = int(y_coords[i]), int(y_coords[i+1])
        x1_l, x2_l = int(left_x[i]), int(left_x[i+1])
        x1_r, x2_r = int(right_x[i]), int(right_x[i+1])

        cv2.line(img, (x1_l, y1), (x2_l, y2), (255, 255, 0), 8)  # Yellow
        cv2.line(img, (x1_r, y1), (x2_r, y2), (255, 255, 255), 8)  # White

    return img

# Test curved detection
curved_image = create_curved_road()
left_fit, right_fit, curvature = detector.detect_curved_lanes(curved_image)

print(f"Left lane polynomial: {left_fit}")
print(f"Right lane polynomial: {right_fit}")
print(f"Radius of curvature: {curvature:.1f} meters")
```

### Integration with Simulation

```python
from av_simulation.core.simulation import Simulation
from av_simulation.perception.lane_detection import StraightLaneDetector

class LaneDetectionSimulation(Simulation):
    def __init__(self):
        super().__init__()
        self.lane_detector = StraightLaneDetector()
        self.detected_lanes = []

    def step(self):
        # Run normal simulation step
        super().step()

        # Detect lanes from current view
        if self.current_env.ego_vehicle:
            # Get vehicle's camera view (simplified)
            camera_image = self.get_camera_view()
            lanes = self.lane_detector.detect_lanes(camera_image)
            self.detected_lanes = lanes

            # Use lane info for navigation
            if lanes:
                lane_center = self.calculate_lane_center(lanes)
                self.adjust_steering_to_lane_center(lane_center)

    def get_camera_view(self):
        # Simplified camera view extraction
        # In real implementation, this would render from vehicle's perspective
        return self.screen
```

## 🧠 Behavioral Planning

### Custom Planning Strategy

```python
from av_simulation.planning.behavioral_planning import BehavioralPlanner
from av_simulation.core.simulation import Simulation

class AggressivePlanner(BehavioralPlanner):
    def __init__(self):
        super().__init__()
        self.risk_tolerance = 0.8  # Higher risk tolerance
        self.target_speed_factor = 1.2  # 20% above speed limit

    def plan_action(self, ego_vehicle, environment):
        state = self.get_current_state(ego_vehicle, environment)

        # Aggressive lane changing
        if state.front_vehicle_distance < 20 and state.left_lane_clear:
            return Action.CHANGE_LANE_LEFT
        elif state.front_vehicle_distance < 20 and state.right_lane_clear:
            return Action.CHANGE_LANE_RIGHT

        # Aggressive acceleration
        if ego_vehicle.state.vx < self.target_speed_factor * environment.speed_limit:
            return Action.ACCELERATE

        return Action.MAINTAIN_SPEED

# Use custom planner
sim = Simulation()
sim.current_env.ego_vehicle.planner = AggressivePlanner()
sim.run()
```

### Learning-Based Planner

```python
import numpy as np
from collections import deque

class LearningPlanner(BehavioralPlanner):
    def __init__(self):
        super().__init__()
        self.experience_buffer = deque(maxlen=1000)
        self.q_table = {}  # Simple Q-learning table
        self.learning_rate = 0.1
        self.epsilon = 0.1  # Exploration rate

    def get_state_key(self, state):
        # Discretize continuous state for Q-table
        front_dist = min(int(state.front_vehicle_distance / 10), 20)
        speed = min(int(state.ego_velocity[0] / 5), 10)
        return (front_dist, speed, state.left_lane_clear, state.right_lane_clear)

    def plan_action(self, ego_vehicle, environment):
        state = self.get_current_state(ego_vehicle, environment)
        state_key = self.get_state_key(state)

        # Initialize Q-values if new state
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in Action}

        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            action = np.random.choice(list(Action))
        else:
            action = max(self.q_table[state_key], key=self.q_table[state_key].get)

        # Store experience for learning
        self.experience_buffer.append({
            'state': state_key,
            'action': action,
            'timestamp': time.time()
        })

        return action

    def update_q_values(self, reward):
        """Update Q-values based on reward signal"""
        if len(self.experience_buffer) < 2:
            return

        # Get last state-action pair
        last_exp = self.experience_buffer[-2]
        current_exp = self.experience_buffer[-1]

        # Q-learning update
        old_q = self.q_table[last_exp['state']][last_exp['action']]
        max_future_q = max(self.q_table[current_exp['state']].values())

        new_q = old_q + self.learning_rate * (reward + 0.9 * max_future_q - old_q)
        self.q_table[last_exp['state']][last_exp['action']] = new_q

# Example usage with learning
sim = Simulation(enable_data_collection=True)
learning_planner = LearningPlanner()
sim.current_env.ego_vehicle.planner = learning_planner

# Run simulation with learning updates
for episode in range(100):
    sim.reset()

    while sim.running:
        sim.step()

        # Calculate reward based on performance
        if sim.current_env.collision_detected:
            reward = -100  # Collision penalty
        elif sim.current_env.ego_vehicle.state.vx > 20:
            reward = 10   # Speed reward
        else:
            reward = 1    # Small positive reward

        learning_planner.update_q_values(reward)

        if sim.current_env.collision_detected:
            break

    print(f"Episode {episode}: Q-table size = {len(learning_planner.q_table)}")
```

## 🛣️ Environment Switching

### Dynamic Environment Changes

```python
from av_simulation.core.simulation import Simulation
import time

# Create simulation
sim = Simulation(enable_data_collection=True)

# Define scenario sequence
scenarios = [
    ('highway', 30),    # Highway for 30 seconds
    ('merge', 20),      # Merge scenario for 20 seconds
    ('roundabout', 25), # Roundabout for 25 seconds
]

for env_type, duration in scenarios:
    print(f"Switching to {env_type} environment for {duration} seconds")

    sim.switch_environment(env_type)

    start_time = time.time()
    while time.time() - start_time < duration:
        sim.step()
        time.sleep(1/60)

    # Collect metrics for this environment
    metrics = sim.get_performance_metrics()
    print(f"{env_type} metrics: {metrics}")

# Export all collected data
sim.data_repository.export_to_csv("multi_environment_data.csv")
```

### Environment-Specific Configuration

```python
from av_simulation.core.simulation import Simulation

# Configure different environments
env_configs = {
    'highway': {
        'traffic_density': 0.3,
        'speed_limit': 35,
        'num_lanes': 4
    },
    'merge': {
        'traffic_density': 0.5,
        'merge_length': 200,
        'merge_traffic_rate': 0.2
    },
    'roundabout': {
        'entry_rate': 0.1,
        'radius': 50,
        'num_exits': 4
    }
}

# Apply configurations
sim = Simulation()
for env_type, config in env_configs.items():
    sim.switch_environment(env_type)
    sim.current_env.configure(config)

    print(f"Running {env_type} with config: {config}")
    # Run for a fixed time...
```

## 🎯 Custom Scenarios

### Creating a Traffic Jam Scenario

```python
from av_simulation.core.simulation import Simulation
from av_simulation.environments.highway import HighwayEnvironment

class TrafficJamScenario(HighwayEnvironment):
    def __init__(self):
        super().__init__()
        self.jam_active = False
        self.jam_start_time = None

    def step(self, dt):
        super().step(dt)

        # Trigger traffic jam after 30 seconds
        if not self.jam_active and time.time() - self.start_time > 30:
            self.create_traffic_jam()
            self.jam_active = True

    def create_traffic_jam(self):
        """Create a traffic jam by slowing down vehicles"""
        print("🚦 Creating traffic jam...")

        for vehicle in self.traffic_vehicles:
            if 500 < vehicle.state.x < 800:  # Jam zone
                vehicle.target_speed = 5  # Slow to 5 m/s
                vehicle.max_acceleration = 1  # Limited acceleration

# Use custom scenario
sim = Simulation()
sim.environments['traffic_jam'] = TrafficJamScenario()
sim.switch_environment('traffic_jam')
sim.run()
```

### Emergency Vehicle Scenario

```python
from av_simulation.vehicles.traffic_vehicle import TrafficVehicle

class EmergencyVehicle(TrafficVehicle):
    def __init__(self, x, y, lane, speed):
        super().__init__(x, y, lane, speed)
        self.is_emergency = True
        self.siren_active = True
        self.color = (255, 0, 0)  # Red for emergency

    def step(self, dt, environment):
        # Emergency vehicles can exceed speed limits
        self.target_speed = min(50, self.max_speed)

        # Request lane changes from other vehicles
        self.request_path_clearing(environment)

        super().step(dt, environment)

    def request_path_clearing(self, environment):
        """Request other vehicles to change lanes"""
        for vehicle in environment.traffic_vehicles:
            if (abs(vehicle.state.x - self.state.x) < 100 and
                vehicle.lane == self.lane):
                # Signal other vehicle to change lanes
                vehicle.emergency_yield = True

# Add emergency vehicle to simulation
sim = Simulation()
sim.switch_environment('highway')

# Add emergency vehicle
emergency = EmergencyVehicle(x=0, y=100, lane=1, speed=40)
sim.current_env.traffic_vehicles.append(emergency)

sim.run()
```

## 📈 Performance Analysis

### Real-time Performance Monitoring

```python
from av_simulation.core.simulation import Simulation
import matplotlib.pyplot as plt
from collections import deque

class PerformanceMonitor:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.collision_rate = deque(maxlen=window_size)
        self.avg_speed = deque(maxlen=window_size)
        self.traffic_flow = deque(maxlen=window_size)

    def update(self, simulation):
        # Calculate metrics
        collisions = len(simulation.current_env.collision_events)
        speeds = [v.state.vx for v in simulation.current_env.all_vehicles()]

        self.collision_rate.append(collisions)
        self.avg_speed.append(sum(speeds) / len(speeds) if speeds else 0)
        self.traffic_flow.append(len(speeds))

    def plot_realtime(self):
        if len(self.avg_speed) < 10:
            return

        plt.clf()

        plt.subplot(3, 1, 1)
        plt.plot(list(self.collision_rate), 'r-')
        plt.title('Collision Rate')
        plt.ylabel('Collisions')

        plt.subplot(3, 1, 2)
        plt.plot(list(self.avg_speed), 'b-')
        plt.title('Average Speed')
        plt.ylabel('Speed (m/s)')

        plt.subplot(3, 1, 3)
        plt.plot(list(self.traffic_flow), 'g-')
        plt.title('Traffic Flow')
        plt.ylabel('Vehicles')

        plt.tight_layout()
        plt.pause(0.01)

# Use performance monitor
sim = Simulation()
monitor = PerformanceMonitor()

plt.ion()  # Interactive plotting
fig = plt.figure(figsize=(10, 8))

try:
    while sim.running:
        sim.step()
        monitor.update(sim)

        # Update plots every 30 frames
        if sim.frame_count % 30 == 0:
            monitor.plot_realtime()

except KeyboardInterrupt:
    print("Simulation stopped by user")

plt.ioff()
plt.show()
```

### Batch Performance Analysis

```python
from av_simulation.core.simulation import Simulation
import pandas as pd
import numpy as np

def run_performance_study():
    """Run multiple simulations with different parameters"""

    results = []

    # Test different traffic densities
    densities = [0.2, 0.3, 0.4, 0.5, 0.6]

    for density in densities:
        print(f"Testing traffic density: {density}")

        # Run 5 simulations for each density
        for run in range(5):
            sim = Simulation(headless=True, enable_data_collection=True)
            sim.current_env.configure({'traffic_density': density})

            # Run for 60 seconds
            start_time = time.time()
            while time.time() - start_time < 60:
                sim.step()

            # Collect metrics
            data = sim.data_repository.get_all_data()

            metrics = {
                'density': density,
                'run': run,
                'collisions': len(data.collision_events),
                'avg_speed': np.mean([s.speed for s in data.vehicle_snapshots]),
                'total_distance': sum([s.position_x for s in data.vehicle_snapshots if s.vehicle_id == 'ego']),
                'lane_changes': len([a for a in data.action_events if 'LANE' in a.action_type])
            }

            results.append(metrics)
            print(f"  Run {run+1}: {metrics['collisions']} collisions, {metrics['avg_speed']:.1f} avg speed")

    # Analyze results
    df = pd.DataFrame(results)

    # Group by density and calculate statistics
    summary = df.groupby('density').agg({
        'collisions': ['mean', 'std'],
        'avg_speed': ['mean', 'std'],
        'total_distance': ['mean', 'std'],
        'lane_changes': ['mean', 'std']
    }).round(2)

    print("\nPerformance Study Results:")
    print(summary)

    # Save results
    df.to_csv('performance_study_results.csv', index=False)
    summary.to_csv('performance_study_summary.csv')

    return df, summary

# Run the study
results_df, summary_df = run_performance_study()
```

## 🔗 More Examples

For additional examples and advanced usage:

- [**Jupyter Notebooks**](notebooks.html) - Interactive examples with visualization
- [**API Reference**](api.html) - Detailed function documentation
- [**GitHub Repository**](https://github.com/aanshshah/av-simulation/tree/main/examples) - Complete example scripts

---

**Next Steps:**
- Try the [Interactive Notebooks](notebooks.html) for hands-on learning
- Read the [API Documentation](api.html) for detailed reference
- Check the [GitHub Issues](https://github.com/aanshshah/av-simulation/issues) for community examples
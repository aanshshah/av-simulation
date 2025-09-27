"""
Utility functions for testing
"""

import numpy as np
import torch
import cv2
import sys
import os
from typing import List, Tuple, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from av_simulation.core.simulation import Vehicle, VehicleState
from av_simulation.planning.behavioral_planning import VehicleState as PlanningVehicleState


class TestUtils:
    """Collection of utility functions for testing"""

    @staticmethod
    def assert_vehicle_states_equal(state1: VehicleState, state2: VehicleState, tolerance: float = 1e-6):
        """Assert that two vehicle states are equal within tolerance"""
        assert abs(state1.x - state2.x) < tolerance, f"x values differ: {state1.x} vs {state2.x}"
        assert abs(state1.y - state2.y) < tolerance, f"y values differ: {state1.y} vs {state2.y}"
        assert abs(state1.angle - state2.angle) < tolerance, f"angle values differ: {state1.angle} vs {state2.angle}"
        assert abs(state1.speed - state2.speed) < tolerance, f"speed values differ: {state1.speed} vs {state2.speed}"

    @staticmethod
    def assert_planning_states_equal(state1: PlanningVehicleState, state2: PlanningVehicleState, tolerance: float = 1e-6):
        """Assert that two planning vehicle states are equal within tolerance"""
        assert abs(state1.x - state2.x) < tolerance, f"x values differ: {state1.x} vs {state2.x}"
        assert abs(state1.y - state2.y) < tolerance, f"y values differ: {state1.y} vs {state2.y}"
        assert abs(state1.vx - state2.vx) < tolerance, f"vx values differ: {state1.vx} vs {state2.vx}"
        assert abs(state1.vy - state2.vy) < tolerance, f"vy values differ: {state1.vy} vs {state2.vy}"
        assert abs(state1.heading - state2.heading) < tolerance, f"heading values differ: {state1.heading} vs {state2.heading}"
        assert abs(state1.angular_velocity - state2.angular_velocity) < tolerance

    @staticmethod
    def create_test_image(width: int = 300, height: int = 200, pattern: str = "lanes") -> np.ndarray:
        """Create a test image with specified pattern"""
        image = np.zeros((height, width, 3), dtype=np.uint8)

        if pattern == "lanes":
            # Add road surface
            image[height//2:, :] = [50, 50, 50]
            # Add lane lines
            cv2.line(image, (width//3, height), (width//3, height//2), (255, 255, 255), 3)
            cv2.line(image, (2*width//3, height), (2*width//3, height//2), (255, 255, 255), 3)

        elif pattern == "curved_lanes":
            # Add curved lanes
            for y in range(height//2, height):
                x_left = int(width//3 + 20 * np.sin(y * 0.1))
                x_right = int(2*width//3 + 20 * np.sin(y * 0.1))
                if 0 <= x_left < width:
                    image[y, max(0, x_left-2):min(width, x_left+3)] = [255, 255, 255]
                if 0 <= x_right < width:
                    image[y, max(0, x_right-2):min(width, x_right+3)] = [255, 255, 255]

        elif pattern == "empty":
            # Keep as black image
            pass

        elif pattern == "noise":
            # Add random noise
            image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

        return image

    @staticmethod
    def create_test_trajectory(length: int = 10, noise_level: float = 0.1) -> List[PlanningVehicleState]:
        """Create a test trajectory with optional noise"""
        trajectory = []
        for i in range(length):
            x = i * 2.0 + np.random.normal(0, noise_level)
            y = np.sin(i * 0.1) * 10 + np.random.normal(0, noise_level)
            vx = 20.0 + np.random.normal(0, noise_level)
            vy = np.random.normal(0, noise_level * 0.5)
            heading = i * 0.05 + np.random.normal(0, noise_level * 0.1)
            angular_velocity = np.random.normal(0, noise_level * 0.01)

            state = PlanningVehicleState(
                x=x, y=y, vx=vx, vy=vy,
                heading=heading, angular_velocity=angular_velocity
            )
            trajectory.append(state)

        return trajectory

    @staticmethod
    def calculate_trajectory_metrics(trajectory: List[PlanningVehicleState]) -> dict:
        """Calculate metrics for a trajectory"""
        if not trajectory:
            return {}

        positions = [(state.x, state.y) for state in trajectory]
        speeds = [np.sqrt(state.vx**2 + state.vy**2) for state in trajectory]

        total_distance = 0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_distance += np.sqrt(dx**2 + dy**2)

        return {
            'total_distance': total_distance,
            'average_speed': np.mean(speeds),
            'max_speed': np.max(speeds),
            'min_speed': np.min(speeds),
            'speed_variance': np.var(speeds),
            'length': len(trajectory)
        }

    @staticmethod
    def validate_image(image: np.ndarray, expected_shape: Tuple[int, ...] = None) -> bool:
        """Validate that an image has correct properties"""
        if not isinstance(image, np.ndarray):
            return False

        if len(image.shape) not in [2, 3]:
            return False

        if expected_shape and image.shape != expected_shape:
            return False

        if image.dtype != np.uint8:
            return False

        return True

    @staticmethod
    def create_collision_scenario() -> Tuple[Vehicle, Vehicle]:
        """Create two vehicles in a collision scenario"""
        vehicle1 = Vehicle(x=100, y=100, angle=0, speed=20)
        vehicle2 = Vehicle(x=105, y=105, angle=np.pi, speed=15)  # Heading towards vehicle1
        return vehicle1, vehicle2

    @staticmethod
    def create_safe_scenario() -> Tuple[Vehicle, Vehicle]:
        """Create two vehicles in a safe scenario"""
        vehicle1 = Vehicle(x=100, y=100, angle=0, speed=20)
        vehicle2 = Vehicle(x=200, y=200, angle=0, speed=20)  # Far away, same direction
        return vehicle1, vehicle2

    @staticmethod
    def simulate_vehicle_interaction(vehicle1: Vehicle, vehicle2: Vehicle, steps: int = 10, dt: float = 0.1) -> List[Tuple[Vehicle, Vehicle]]:
        """Simulate interaction between two vehicles"""
        history = []

        for _ in range(steps):
            # Store current state
            v1_copy = Vehicle(vehicle1.x, vehicle1.y, vehicle1.angle, vehicle1.speed)
            v2_copy = Vehicle(vehicle2.x, vehicle2.y, vehicle2.angle, vehicle2.speed)
            history.append((v1_copy, v2_copy))

            # Update vehicles
            vehicle1.update(dt)
            vehicle2.update(dt)

        return history

    @staticmethod
    def tensor_almost_equal(tensor1: torch.Tensor, tensor2: torch.Tensor, tolerance: float = 1e-6) -> bool:
        """Check if two tensors are almost equal"""
        if tensor1.shape != tensor2.shape:
            return False
        return torch.allclose(tensor1, tensor2, atol=tolerance)

    @staticmethod
    def count_lane_pixels(image: np.ndarray, lane_color: Tuple[int, int, int] = (255, 255, 255)) -> int:
        """Count pixels of lane color in image"""
        if len(image.shape) == 3:
            mask = np.all(image == lane_color, axis=2)
        else:
            mask = image == lane_color[0]  # Assume grayscale
        return np.sum(mask)

    @staticmethod
    def generate_performance_metrics(execution_times: List[float]) -> dict:
        """Generate performance metrics from execution times"""
        if not execution_times:
            return {}

        return {
            'mean_time': np.mean(execution_times),
            'median_time': np.median(execution_times),
            'std_time': np.std(execution_times),
            'min_time': np.min(execution_times),
            'max_time': np.max(execution_times),
            'total_time': np.sum(execution_times),
            'count': len(execution_times)
        }

    @staticmethod
    def create_benchmark_data(size: int = 1000) -> dict:
        """Create benchmark data for performance testing"""
        return {
            'states': [
                PlanningVehicleState(
                    x=np.random.uniform(0, 1000),
                    y=np.random.uniform(0, 1000),
                    vx=np.random.uniform(10, 30),
                    vy=np.random.uniform(-5, 5),
                    heading=np.random.uniform(0, 2*np.pi),
                    angular_velocity=np.random.uniform(-0.1, 0.1)
                )
                for _ in range(size)
            ],
            'actions': np.random.randint(0, 8, size),
            'rewards': np.random.uniform(-1, 1, size),
            'images': [TestUtils.create_test_image() for _ in range(min(100, size))]
        }

    @staticmethod
    def validate_model_output(output: torch.Tensor, expected_shape: Tuple[int, ...], value_range: Tuple[float, float] = None) -> bool:
        """Validate neural network model output"""
        if not isinstance(output, torch.Tensor):
            return False

        if output.shape != expected_shape:
            return False

        if value_range:
            min_val, max_val = value_range
            if torch.min(output) < min_val or torch.max(output) > max_val:
                return False

        if torch.isnan(output).any() or torch.isinf(output).any():
            return False

        return True

    @staticmethod
    def setup_test_environment() -> dict:
        """Setup test environment with common configurations"""
        return {
            'numpy_seed': 42,
            'torch_seed': 42,
            'test_image_size': (200, 300, 3),
            'test_trajectory_length': 10,
            'default_tolerance': 1e-6,
            'performance_threshold': 0.1  # seconds
        }
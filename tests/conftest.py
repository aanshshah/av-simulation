"""
Pytest configuration and shared fixtures for av_simulation tests
"""

import pytest
import numpy as np
import torch
import pygame
import cv2
import sys
import os
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from av_simulation.core.simulation import Vehicle, VehicleState, HighwayEnvironment
from av_simulation.detection.lane_detection import StraightLaneDetector, CurvedLaneDetector
from av_simulation.planning.behavioral_planning import VehicleState as PlanningVehicleState


@pytest.fixture(scope="session", autouse=True)
def setup_pygame():
    """Initialize pygame for all tests that need it"""
    pygame.init()
    yield
    pygame.quit()


@pytest.fixture(scope="session")
def set_random_seeds():
    """Set random seeds for reproducible tests"""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def sample_vehicle():
    """Create a sample vehicle for testing"""
    return Vehicle(x=100, y=100, angle=0, speed=20)


@pytest.fixture
def sample_vehicle_state():
    """Create a sample vehicle state for testing"""
    return VehicleState(x=100.0, y=200.0, angle=0.5, speed=25.0)


@pytest.fixture
def sample_planning_vehicle_state():
    """Create a sample planning vehicle state for testing"""
    return PlanningVehicleState(
        x=100.0, y=200.0, vx=25.0, vy=5.0,
        heading=0.5, angular_velocity=0.1
    )


@pytest.fixture
def highway_environment():
    """Create a highway environment for testing"""
    return HighwayEnvironment(width=800, height=600)


@pytest.fixture
def straight_lane_detector():
    """Create a straight lane detector for testing"""
    return StraightLaneDetector()


@pytest.fixture
def curved_lane_detector():
    """Create a curved lane detector for testing"""
    return CurvedLaneDetector()


@pytest.fixture
def sample_road_image():
    """Create a synthetic road image for testing"""
    height, width = 200, 300
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Add road surface
    image[height//2:, :] = [50, 50, 50]  # Gray road

    # Add white lane lines
    cv2.line(image, (width//3, height), (width//3, height//2), (255, 255, 255), 3)
    cv2.line(image, (2*width//3, height), (2*width//3, height//2), (255, 255, 255), 3)

    return image


@pytest.fixture
def sample_curved_road_image():
    """Create a synthetic curved road image for testing"""
    height, width = 200, 300
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Add curved lane markings
    for y in range(height//2, height):
        x_left = int(width//3 + 20 * np.sin(y * 0.1))
        x_right = int(2*width//3 + 20 * np.sin(y * 0.1))
        if 0 <= x_left < width:
            image[y, max(0, x_left-2):min(width, x_left+3)] = [255, 255, 255]
        if 0 <= x_right < width:
            image[y, max(0, x_right-2):min(width, x_right+3)] = [255, 255, 255]

    return image


@pytest.fixture
def sample_binary_image():
    """Create a binary image for lane detection testing"""
    height, width = 400, 600
    binary = np.zeros((height, width), dtype=np.uint8)

    # Add vertical lines representing lane markings
    binary[200:, 150:153] = 255  # Left line
    binary[200:, 450:453] = 255  # Right line

    return binary


@pytest.fixture
def mock_cv2_functions():
    """Mock OpenCV functions for testing without actual image processing"""
    with patch('cv2.findChessboardCorners') as mock_find, \
         patch('cv2.calibrateCamera') as mock_calibrate, \
         patch('cv2.imshow') as mock_imshow, \
         patch('cv2.waitKey') as mock_waitkey, \
         patch('cv2.destroyAllWindows') as mock_destroy:

        # Configure mocks
        mock_find.return_value = (True, np.random.rand(54, 1, 2))
        mock_calibrate.return_value = (
            1.0,  # ret
            np.eye(3),  # mtx
            np.zeros(5),  # dist
            None, None  # rvecs, tvecs
        )
        mock_waitkey.return_value = ord('q')

        yield {
            'find': mock_find,
            'calibrate': mock_calibrate,
            'imshow': mock_imshow,
            'waitkey': mock_waitkey,
            'destroy': mock_destroy
        }


@pytest.fixture
def mock_pygame_display():
    """Mock pygame display functions for headless testing"""
    with patch('pygame.display.set_mode') as mock_set_mode, \
         patch('pygame.display.flip') as mock_flip, \
         patch('pygame.display.set_caption') as mock_caption:

        # Create mock surface
        mock_surface = Mock()
        mock_surface.fill = Mock()
        mock_surface.blit = Mock()
        mock_surface.get_rect = Mock(return_value=pygame.Rect(0, 0, 800, 600))

        mock_set_mode.return_value = mock_surface

        yield {
            'set_mode': mock_set_mode,
            'flip': mock_flip,
            'caption': mock_caption,
            'surface': mock_surface
        }


@pytest.fixture
def sample_experience_data():
    """Create sample experience data for RL testing"""
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    for i in range(100):
        state = PlanningVehicleState(
            x=float(i), y=200.0, vx=25.0, vy=5.0,
            heading=0.5, angular_velocity=0.1
        )
        next_state = PlanningVehicleState(
            x=float(i+1), y=200.0, vx=25.0, vy=5.0,
            heading=0.5, angular_velocity=0.1
        )

        states.append(state)
        actions.append(0)  # Action index
        rewards.append(np.random.rand())
        next_states.append(next_state)
        dones.append(False)

    return {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'next_states': next_states,
        'dones': dones
    }


@pytest.fixture
def temp_calibration_images(tmp_path):
    """Create temporary calibration images for camera calibration testing"""
    images = []
    for i in range(5):
        # Create chessboard pattern
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        for row in range(0, 100, 10):
            for col in range(0, 100, 10):
                if (row // 10 + col // 10) % 2 == 0:
                    img[row:row+10, col:col+10] = 255

        # Save temporary image
        img_path = tmp_path / f"calibration_{i}.jpg"
        cv2.imwrite(str(img_path), img)
        images.append(img)

    return images


class TestDataFactory:
    """Factory class for creating test data"""

    @staticmethod
    def create_vehicle_trajectory(length=10, start_x=0, start_y=0):
        """Create a trajectory of vehicle states"""
        trajectory = []
        for i in range(length):
            state = PlanningVehicleState(
                x=start_x + i * 2.0,
                y=start_y + np.sin(i * 0.1) * 10,
                vx=20.0 + np.random.normal(0, 1),
                vy=np.random.normal(0, 0.5),
                heading=i * 0.05,
                angular_velocity=np.random.normal(0, 0.01)
            )
            trajectory.append(state)
        return trajectory

    @staticmethod
    def create_multi_vehicle_scenario(num_vehicles=5):
        """Create a multi-vehicle scenario"""
        vehicles = []
        for i in range(num_vehicles):
            vehicle = Vehicle(
                x=100 + i * 50,
                y=200 + np.random.normal(0, 10),
                angle=np.random.normal(0, 0.1),
                speed=20 + np.random.normal(0, 5)
            )
            vehicles.append(vehicle)
        return vehicles

    @staticmethod
    def create_lane_detection_dataset(size=50):
        """Create a dataset of synthetic road images"""
        dataset = []
        for i in range(size):
            height, width = 200, 300
            image = np.zeros((height, width, 3), dtype=np.uint8)

            # Add road
            image[height//2:, :] = [50 + np.random.randint(-10, 10), 50, 50]

            # Add lanes with variation
            lane_offset = np.random.randint(-20, 20)
            cv2.line(image,
                    (width//3 + lane_offset, height),
                    (width//3 + lane_offset, height//2),
                    (255, 255, 255), 3)
            cv2.line(image,
                    (2*width//3 + lane_offset, height),
                    (2*width//3 + lane_offset, height//2),
                    (255, 255, 255), 3)

            dataset.append(image)

        return dataset


@pytest.fixture
def test_data_factory():
    """Provide access to test data factory"""
    return TestDataFactory()


# Markers for different test categories
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow
pytest.mark.gpu = pytest.mark.gpu
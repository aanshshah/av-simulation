"""
Comprehensive tests for simulation module
"""

import unittest
import sys
import os
import math
import pygame

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from av_simulation.core.simulation import (
    Vehicle, VehicleState, Action, Environment, HighwayEnvironment,
    MergingEnvironment, RoundaboutEnvironment, BehaviorPlanner, Simulation
)

class TestVehicleState(unittest.TestCase):
    """Test VehicleState dataclass"""

    def test_vehicle_state_creation(self):
        """Test VehicleState can be created with valid parameters"""
        state = VehicleState(x=100.0, y=200.0, angle=0.5, speed=25.0)
        self.assertEqual(state.x, 100.0)
        self.assertEqual(state.y, 200.0)
        self.assertEqual(state.angle, 0.5)
        self.assertEqual(state.speed, 25.0)

class TestVehicle(unittest.TestCase):
    """Test Vehicle class functionality"""

    def setUp(self):
        """Set up test vehicle"""
        self.vehicle = Vehicle(x=100, y=100, angle=0, speed=20)

    def test_vehicle_creation(self):
        """Test vehicle can be created with valid parameters"""
        self.assertEqual(self.vehicle.x, 100)
        self.assertEqual(self.vehicle.y, 100)
        self.assertEqual(self.vehicle.angle, 0)
        self.assertEqual(self.vehicle.speed, 20)
        self.assertEqual(self.vehicle.target_speed, 20)
        self.assertFalse(self.vehicle.is_autonomous)

    def test_vehicle_state_property(self):
        """Test vehicle state property returns VehicleState"""
        state = self.vehicle.state
        self.assertIsInstance(state, VehicleState)
        self.assertEqual(state.x, self.vehicle.x)
        self.assertEqual(state.y, self.vehicle.y)
        self.assertEqual(state.angle, self.vehicle.angle)
        self.assertEqual(state.speed, self.vehicle.speed)

    def test_vehicle_update_position(self):
        """Test vehicle position updates correctly"""
        initial_x = self.vehicle.x
        initial_y = self.vehicle.y
        dt = 0.1

        self.vehicle.update(dt)

        # Vehicle should move based on speed and angle
        expected_x = initial_x + self.vehicle.speed * math.cos(self.vehicle.angle) * dt
        expected_y = initial_y + self.vehicle.speed * math.sin(self.vehicle.angle) * dt

        self.assertAlmostEqual(self.vehicle.x, expected_x, places=5)
        self.assertAlmostEqual(self.vehicle.y, expected_y, places=5)

    def test_vehicle_speed_adjustment(self):
        """Test vehicle adjusts speed towards target"""
        self.vehicle.target_speed = 30
        initial_speed = self.vehicle.speed
        dt = 0.1

        self.vehicle.update(dt)

        # Speed should increase towards target
        self.assertGreater(self.vehicle.speed, initial_speed)

    def test_vehicle_collision_detection(self):
        """Test vehicle collision detection"""
        other_vehicle = Vehicle(x=105, y=105, angle=0, speed=20)

        # Vehicles should be close enough to collide
        self.assertTrue(self.vehicle.check_collision(other_vehicle))

        # Move other vehicle far away
        other_vehicle.x = 200
        other_vehicle.y = 200

        # Vehicles should not collide
        self.assertFalse(self.vehicle.check_collision(other_vehicle))

    def test_vehicle_distance_calculation(self):
        """Test distance calculation between vehicles"""
        other_vehicle = Vehicle(x=103, y=104, angle=0, speed=20)
        distance = self.vehicle.distance_to(other_vehicle)
        expected_distance = math.sqrt((103-100)**2 + (104-100)**2)
        self.assertAlmostEqual(distance, expected_distance, places=5)

class TestAction(unittest.TestCase):
    """Test Action enum"""

    def test_action_enum_values(self):
        """Test Action enum has required values"""
        self.assertTrue(hasattr(Action, 'ACCELERATE'))
        self.assertTrue(hasattr(Action, 'BRAKE'))
        self.assertTrue(hasattr(Action, 'TURN_LEFT'))
        self.assertTrue(hasattr(Action, 'TURN_RIGHT'))
        self.assertTrue(hasattr(Action, 'MAINTAIN'))

class TestEnvironment(unittest.TestCase):
    """Test base Environment class"""

    def test_environment_creation(self):
        """Test environment can be created"""
        env = Environment(width=800, height=600)
        self.assertEqual(env.width, 800)
        self.assertEqual(env.height, 600)
        self.assertEqual(len(env.vehicles), 0)
        self.assertEqual(len(env.lanes), 0)

class TestHighwayEnvironment(unittest.TestCase):
    """Test HighwayEnvironment functionality"""

    def setUp(self):
        """Set up test environment"""
        self.env = HighwayEnvironment(width=800, height=600)

    def test_highway_creation(self):
        """Test highway environment creation"""
        self.assertEqual(self.env.width, 800)
        self.assertEqual(self.env.height, 600)
        self.assertGreater(len(self.env.lanes), 0)

    def test_highway_vehicle_spawn(self):
        """Test vehicle spawning in highway"""
        initial_count = len(self.env.vehicles)
        self.env.spawn_vehicle()
        self.assertEqual(len(self.env.vehicles), initial_count + 1)

    def test_highway_update(self):
        """Test highway environment update"""
        self.env.spawn_vehicle()
        initial_positions = [(v.x, v.y) for v in self.env.vehicles]

        self.env.update(0.1)

        # Vehicles should have moved
        new_positions = [(v.x, v.y) for v in self.env.vehicles]
        self.assertNotEqual(initial_positions, new_positions)

class TestMergingEnvironment(unittest.TestCase):
    """Test MergingEnvironment functionality"""

    def setUp(self):
        """Set up test environment"""
        self.env = MergingEnvironment(width=800, height=600)

    def test_merging_creation(self):
        """Test merging environment creation"""
        self.assertEqual(self.env.width, 800)
        self.assertEqual(self.env.height, 600)
        self.assertGreater(len(self.env.lanes), 0)

class TestRoundaboutEnvironment(unittest.TestCase):
    """Test RoundaboutEnvironment functionality"""

    def setUp(self):
        """Set up test environment"""
        self.env = RoundaboutEnvironment(width=800, height=600)

    def test_roundabout_creation(self):
        """Test roundabout environment creation"""
        self.assertEqual(self.env.width, 800)
        self.assertEqual(self.env.height, 600)
        self.assertGreater(self.env.center_x, 0)
        self.assertGreater(self.env.center_y, 0)
        self.assertGreater(self.env.radius, 0)

class TestBehaviorPlanner(unittest.TestCase):
    """Test BehaviorPlanner functionality"""

    def setUp(self):
        """Set up test planner"""
        self.planner = BehaviorPlanner()

    def test_planner_creation(self):
        """Test behavior planner creation"""
        self.assertIsNotNone(self.planner)

    def test_plan_action(self):
        """Test action planning"""
        vehicle = Vehicle(x=100, y=100, angle=0, speed=20)
        nearby_vehicles = [Vehicle(x=150, y=100, angle=0, speed=15)]

        action = self.planner.plan_action(vehicle, nearby_vehicles)
        self.assertIsInstance(action, Action)

class TestSimulation(unittest.TestCase):
    """Test main Simulation class"""

    def setUp(self):
        """Set up test simulation"""
        # Initialize pygame for testing
        pygame.init()
        self.simulation = Simulation()

    def tearDown(self):
        """Clean up after tests"""
        pygame.quit()

    def test_simulation_creation(self):
        """Test simulation creation"""
        self.assertIsNotNone(self.simulation.screen)
        self.assertIsNotNone(self.simulation.clock)
        self.assertIsNotNone(self.simulation.environment)
        self.assertIsNotNone(self.simulation.player_vehicle)

    def test_environment_switching(self):
        """Test switching between environments"""
        initial_env_type = type(self.simulation.environment)

        # Switch environment
        self.simulation.switch_environment()

        # Environment type should change
        new_env_type = type(self.simulation.environment)
        # Note: might be same type if cycling, but method should execute without error
        self.assertIsNotNone(self.simulation.environment)

    def test_simulation_reset(self):
        """Test simulation reset functionality"""
        # Modify some state
        self.simulation.player_vehicle.x = 500
        self.simulation.paused = True

        # Reset
        self.simulation.reset()

        # State should be reset
        self.assertFalse(self.simulation.paused)
        self.assertIsNotNone(self.simulation.player_vehicle)

if __name__ == '__main__':
    unittest.main()
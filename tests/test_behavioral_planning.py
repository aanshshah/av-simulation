"""
Comprehensive tests for behavioral planning module
"""

import unittest
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from av_simulation.planning.behavioral_planning import (
    VehicleAction, VehicleState, DynamicsModel, ExperienceBuffer,
    ModelBasedRL, CrossEntropyMethod, RobustControl, POMDP
)

class TestVehicleAction(unittest.TestCase):
    """Test VehicleAction enum"""

    def test_action_enum_values(self):
        """Test VehicleAction enum has required values"""
        self.assertTrue(hasattr(VehicleAction, 'ACCELERATE'))
        self.assertTrue(hasattr(VehicleAction, 'BRAKE'))
        self.assertTrue(hasattr(VehicleAction, 'TURN_LEFT'))
        self.assertTrue(hasattr(VehicleAction, 'TURN_RIGHT'))
        self.assertTrue(hasattr(VehicleAction, 'MAINTAIN_SPEED'))
        self.assertTrue(hasattr(VehicleAction, 'LANE_CHANGE_LEFT'))
        self.assertTrue(hasattr(VehicleAction, 'LANE_CHANGE_RIGHT'))
        self.assertTrue(hasattr(VehicleAction, 'EMERGENCY_BRAKE'))

class TestVehicleState(unittest.TestCase):
    """Test VehicleState dataclass"""

    def test_vehicle_state_creation(self):
        """Test VehicleState can be created with valid parameters"""
        state = VehicleState(x=100.0, y=200.0, vx=25.0, vy=5.0,
                           heading=0.5, angular_velocity=0.1)
        self.assertEqual(state.x, 100.0)
        self.assertEqual(state.y, 200.0)
        self.assertEqual(state.vx, 25.0)
        self.assertEqual(state.vy, 5.0)
        self.assertEqual(state.heading, 0.5)
        self.assertEqual(state.angular_velocity, 0.1)

    def test_vehicle_state_to_array(self):
        """Test VehicleState conversion to numpy array"""
        state = VehicleState(x=100.0, y=200.0, vx=25.0, vy=5.0,
                           heading=0.5, angular_velocity=0.1)
        array = state.to_array()

        expected = np.array([100.0, 200.0, 25.0, 5.0, 0.5, 0.1])
        np.testing.assert_array_equal(array, expected)

    def test_vehicle_state_from_array(self):
        """Test VehicleState creation from numpy array"""
        array = np.array([100.0, 200.0, 25.0, 5.0, 0.5, 0.1])
        state = VehicleState.from_array(array)

        self.assertEqual(state.x, 100.0)
        self.assertEqual(state.y, 200.0)
        self.assertEqual(state.vx, 25.0)
        self.assertEqual(state.vy, 5.0)
        self.assertEqual(state.heading, 0.5)
        self.assertEqual(state.angular_velocity, 0.1)

class TestDynamicsModel(unittest.TestCase):
    """Test DynamicsModel neural network"""

    def setUp(self):
        """Set up test model"""
        self.model = DynamicsModel(state_dim=6, action_dim=8, hidden_dim=64)

    def test_model_creation(self):
        """Test dynamics model creation"""
        self.assertIsInstance(self.model, nn.Module)
        self.assertEqual(self.model.state_dim, 6)
        self.assertEqual(self.model.action_dim, 8)
        self.assertEqual(self.model.hidden_dim, 64)

    def test_model_forward_pass(self):
        """Test model forward pass"""
        batch_size = 10
        state = torch.randn(batch_size, 6)
        action = torch.randn(batch_size, 8)

        next_state = self.model(state, action)

        # Output should have same batch size and state dimension
        self.assertEqual(next_state.shape, (batch_size, 6))

    def test_model_predict_single(self):
        """Test single state prediction"""
        current_state = VehicleState(x=100.0, y=200.0, vx=25.0, vy=5.0,
                                   heading=0.5, angular_velocity=0.1)
        action = VehicleAction.ACCELERATE

        next_state = self.model.predict(current_state, action)

        self.assertIsInstance(next_state, VehicleState)

    def test_model_batch_predict(self):
        """Test batch prediction"""
        states = [
            VehicleState(x=100.0, y=200.0, vx=25.0, vy=5.0, heading=0.5, angular_velocity=0.1),
            VehicleState(x=200.0, y=300.0, vx=30.0, vy=0.0, heading=0.0, angular_velocity=0.0)
        ]
        actions = [VehicleAction.ACCELERATE, VehicleAction.BRAKE]

        next_states = self.model.predict_batch(states, actions)

        self.assertEqual(len(next_states), 2)
        for state in next_states:
            self.assertIsInstance(state, VehicleState)

class TestExperienceBuffer(unittest.TestCase):
    """Test ExperienceBuffer functionality"""

    def setUp(self):
        """Set up test buffer"""
        self.buffer = ExperienceBuffer(capacity=100)

    def test_buffer_creation(self):
        """Test experience buffer creation"""
        self.assertEqual(self.buffer.capacity, 100)
        self.assertEqual(len(self.buffer), 0)

    def test_buffer_add_experience(self):
        """Test adding experience to buffer"""
        state = VehicleState(x=100.0, y=200.0, vx=25.0, vy=5.0,
                           heading=0.5, angular_velocity=0.1)
        action = VehicleAction.ACCELERATE
        reward = 1.0
        next_state = VehicleState(x=102.0, y=201.0, vx=26.0, vy=5.0,
                                heading=0.5, angular_velocity=0.1)
        done = False

        self.buffer.add(state, action, reward, next_state, done)

        self.assertEqual(len(self.buffer), 1)

    def test_buffer_overflow(self):
        """Test buffer behavior when capacity is exceeded"""
        # Fill buffer beyond capacity
        for i in range(150):
            state = VehicleState(x=float(i), y=200.0, vx=25.0, vy=5.0,
                               heading=0.5, angular_velocity=0.1)
            self.buffer.add(state, VehicleAction.MAINTAIN_SPEED, 0.0, state, False)

        # Buffer should not exceed capacity
        self.assertEqual(len(self.buffer), 100)

    def test_buffer_sample(self):
        """Test sampling from buffer"""
        # Add some experiences
        for i in range(50):
            state = VehicleState(x=float(i), y=200.0, vx=25.0, vy=5.0,
                               heading=0.5, angular_velocity=0.1)
            self.buffer.add(state, VehicleAction.MAINTAIN_SPEED, 0.0, state, False)

        # Sample batch
        batch = self.buffer.sample(batch_size=10)

        self.assertEqual(len(batch), 5)  # states, actions, rewards, next_states, dones
        self.assertEqual(len(batch[0]), 10)  # batch size

class TestModelBasedRL(unittest.TestCase):
    """Test ModelBasedRL functionality"""

    def setUp(self):
        """Set up test MRL agent"""
        self.agent = ModelBasedRL(state_dim=6, action_dim=8, lr=0.001)

    def test_agent_creation(self):
        """Test MRL agent creation"""
        self.assertIsNotNone(self.agent.dynamics_model)
        self.assertIsNotNone(self.agent.policy_net)
        self.assertIsNotNone(self.agent.value_net)
        self.assertIsNotNone(self.agent.experience_buffer)

    def test_action_selection(self):
        """Test action selection"""
        state = VehicleState(x=100.0, y=200.0, vx=25.0, vy=5.0,
                           heading=0.5, angular_velocity=0.1)

        action = self.agent.select_action(state)

        self.assertIsInstance(action, VehicleAction)

    def test_model_rollout(self):
        """Test model-based rollout"""
        initial_state = VehicleState(x=100.0, y=200.0, vx=25.0, vy=5.0,
                                   heading=0.5, angular_velocity=0.1)
        horizon = 5

        trajectory = self.agent.rollout(initial_state, horizon)

        self.assertEqual(len(trajectory), horizon + 1)  # Including initial state
        for state in trajectory:
            self.assertIsInstance(state, VehicleState)

    def test_update_policy(self):
        """Test policy update"""
        # Add some experiences to buffer first
        for i in range(50):
            state = VehicleState(x=float(i), y=200.0, vx=25.0, vy=5.0,
                               heading=0.5, angular_velocity=0.1)
            self.agent.experience_buffer.add(
                state, VehicleAction.MAINTAIN_SPEED, 0.0, state, False
            )

        # Update should run without errors
        initial_loss = self.agent.update_policy()
        self.assertIsInstance(initial_loss, (int, float))

class TestCrossEntropyMethod(unittest.TestCase):
    """Test CrossEntropyMethod optimization"""

    def setUp(self):
        """Set up test CEM optimizer"""
        self.cem = CrossEntropyMethod(
            action_dim=8,
            population_size=50,
            elite_frac=0.2,
            max_iterations=10
        )

    def test_cem_creation(self):
        """Test CEM optimizer creation"""
        self.assertEqual(self.cem.action_dim, 8)
        self.assertEqual(self.cem.population_size, 50)
        self.assertEqual(self.cem.elite_frac, 0.2)
        self.assertEqual(self.cem.max_iterations, 10)

    def test_optimize_simple_function(self):
        """Test CEM optimization on simple quadratic function"""
        def objective_function(x):
            # Simple quadratic with minimum at origin
            return -np.sum(x**2, axis=1)

        best_action = self.cem.optimize(objective_function)

        self.assertEqual(len(best_action), 8)
        # Should be close to zero (global minimum)
        self.assertTrue(np.allclose(best_action, 0, atol=1.0))

    def test_optimize_action_sequence(self):
        """Test optimizing action sequence for vehicle"""
        def vehicle_objective(action_sequence):
            # Mock objective: prefer moderate actions
            scores = []
            for actions in action_sequence:
                # Penalize extreme actions
                score = -np.sum(np.abs(actions))
                scores.append(score)
            return np.array(scores)

        initial_state = VehicleState(x=100.0, y=200.0, vx=25.0, vy=5.0,
                                   heading=0.5, angular_velocity=0.1)
        horizon = 5

        best_actions = self.cem.optimize_action_sequence(initial_state, horizon, vehicle_objective)

        self.assertEqual(len(best_actions), horizon)
        for action_vec in best_actions:
            self.assertEqual(len(action_vec), 8)

class TestRobustControl(unittest.TestCase):
    """Test RobustControl functionality"""

    def setUp(self):
        """Set up test robust controller"""
        self.controller = RobustControl(uncertainty_threshold=0.1, safety_margin=2.0)

    def test_controller_creation(self):
        """Test robust controller creation"""
        self.assertEqual(self.controller.uncertainty_threshold, 0.1)
        self.assertEqual(self.controller.safety_margin, 2.0)

    def test_uncertainty_estimation(self):
        """Test uncertainty estimation"""
        observations = [
            VehicleState(x=100.0 + i, y=200.0, vx=25.0, vy=5.0,
                        heading=0.5, angular_velocity=0.1)
            for i in range(10)
        ]

        uncertainty = self.controller.estimate_uncertainty(observations)

        self.assertIsInstance(uncertainty, (int, float))
        self.assertGreaterEqual(uncertainty, 0)

    def test_robust_action_selection(self):
        """Test robust action selection"""
        current_state = VehicleState(x=100.0, y=200.0, vx=25.0, vy=5.0,
                                   heading=0.5, angular_velocity=0.1)
        nearby_vehicles = [
            VehicleState(x=120.0, y=200.0, vx=20.0, vy=0.0,
                        heading=0.0, angular_velocity=0.0)
        ]

        action = self.controller.robust_action_selection(current_state, nearby_vehicles)

        self.assertIsInstance(action, VehicleAction)

    def test_continuous_ambiguity_prediction(self):
        """Test continuous ambiguity prediction"""
        current_observation = VehicleState(x=100.0, y=200.0, vx=25.0, vy=5.0,
                                         heading=0.5, angular_velocity=0.1)
        observation_history = [
            VehicleState(x=90.0 + i, y=200.0, vx=25.0, vy=5.0,
                        heading=0.5, angular_velocity=0.1)
            for i in range(10)
        ]

        predicted_trajectory = self.controller.continuous_ambiguity_prediction(
            current_observation, observation_history, horizon=5
        )

        self.assertEqual(len(predicted_trajectory), 5)
        for state in predicted_trajectory:
            self.assertIsInstance(state, VehicleState)

class TestPOMDP(unittest.TestCase):
    """Test POMDP functionality"""

    def setUp(self):
        """Set up test POMDP"""
        self.pomdp = POMDP(state_dim=6, action_dim=8, observation_dim=4)

    def test_pomdp_creation(self):
        """Test POMDP creation"""
        self.assertEqual(self.pomdp.state_dim, 6)
        self.assertEqual(self.pomdp.action_dim, 8)
        self.assertEqual(self.pomdp.observation_dim, 4)

    def test_belief_update(self):
        """Test belief state update"""
        # Create mock observation
        observation = np.random.randn(4)
        action = VehicleAction.ACCELERATE

        # Update belief
        self.pomdp.update_belief(observation, action)

        # Belief should be updated (not None)
        self.assertIsNotNone(self.pomdp.current_belief)

    def test_belief_prediction(self):
        """Test belief-based action prediction"""
        # Initialize with some belief
        self.pomdp.current_belief = np.random.randn(6)

        action = self.pomdp.predict_action()

        self.assertIsInstance(action, VehicleAction)

    def test_policy_evaluation(self):
        """Test policy evaluation in POMDP"""
        # Create test trajectory
        trajectory = [
            (np.random.randn(6), VehicleAction.ACCELERATE, 1.0),
            (np.random.randn(6), VehicleAction.MAINTAIN_SPEED, 0.5),
            (np.random.randn(6), VehicleAction.BRAKE, -0.5)
        ]

        value = self.pomdp.evaluate_policy(trajectory)

        self.assertIsInstance(value, (int, float))

class TestIntegration(unittest.TestCase):
    """Integration tests for behavioral planning components"""

    def test_complete_planning_pipeline(self):
        """Test complete behavioral planning pipeline"""
        # Initialize components
        mrl_agent = ModelBasedRL(state_dim=6, action_dim=8)
        cem_optimizer = CrossEntropyMethod(action_dim=8, population_size=20, max_iterations=5)
        robust_controller = RobustControl()
        pomdp = POMDP(state_dim=6, action_dim=8, observation_dim=4)

        # Create scenario
        current_state = VehicleState(x=100.0, y=200.0, vx=25.0, vy=5.0,
                                   heading=0.5, angular_velocity=0.1)

        # Test MRL action selection
        mrl_action = mrl_agent.select_action(current_state)
        self.assertIsInstance(mrl_action, VehicleAction)

        # Test CEM optimization
        def mock_objective(actions):
            return np.random.randn(len(actions))

        cem_action = cem_optimizer.optimize(mock_objective)
        self.assertEqual(len(cem_action), 8)

        # Test robust control
        nearby_vehicles = [
            VehicleState(x=120.0, y=200.0, vx=20.0, vy=0.0,
                        heading=0.0, angular_velocity=0.0)
        ]
        robust_action = robust_controller.robust_action_selection(current_state, nearby_vehicles)
        self.assertIsInstance(robust_action, VehicleAction)

        # Test POMDP
        observation = np.random.randn(4)
        pomdp.update_belief(observation, VehicleAction.ACCELERATE)
        pomdp_action = pomdp.predict_action()
        self.assertIsInstance(pomdp_action, VehicleAction)

    def test_component_interoperability(self):
        """Test that components can work together"""
        # Test that state representations are compatible
        state = VehicleState(x=100.0, y=200.0, vx=25.0, vy=5.0,
                           heading=0.5, angular_velocity=0.1)

        # Convert to array and back
        state_array = state.to_array()
        reconstructed_state = VehicleState.from_array(state_array)

        # States should be equivalent
        self.assertEqual(state.x, reconstructed_state.x)
        self.assertEqual(state.y, reconstructed_state.y)
        self.assertEqual(state.vx, reconstructed_state.vx)
        self.assertEqual(state.vy, reconstructed_state.vy)
        self.assertEqual(state.heading, reconstructed_state.heading)
        self.assertEqual(state.angular_velocity, reconstructed_state.angular_velocity)

class TestErrorHandling(unittest.TestCase):
    """Test error handling in behavioral planning"""

    def test_invalid_state_dimensions(self):
        """Test handling of invalid state dimensions"""
        with self.assertRaises((ValueError, TypeError)):
            VehicleState.from_array(np.array([1, 2, 3]))  # Wrong dimension

    def test_model_with_invalid_inputs(self):
        """Test dynamics model with invalid inputs"""
        model = DynamicsModel(state_dim=6, action_dim=8)

        with self.assertRaises((ValueError, RuntimeError)):
            # Wrong tensor dimensions
            state = torch.randn(10, 4)  # Should be (10, 6)
            action = torch.randn(10, 8)
            model(state, action)

    def test_buffer_invalid_operations(self):
        """Test experience buffer error handling"""
        buffer = ExperienceBuffer(capacity=10)

        # Try to sample from empty buffer
        with self.assertRaises((ValueError, IndexError)):
            buffer.sample(batch_size=5)

    def test_cem_invalid_parameters(self):
        """Test CEM with invalid parameters"""
        with self.assertRaises((ValueError, TypeError)):
            CrossEntropyMethod(action_dim=0, population_size=10)  # Invalid action_dim

if __name__ == '__main__':
    # Set random seeds for reproducible tests
    np.random.seed(42)
    torch.manual_seed(42)

    unittest.main()
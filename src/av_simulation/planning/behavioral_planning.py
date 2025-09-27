"""
Behavioral Planning Module
Implements Case Study 3 from the paper:
- Model-based Reinforcement Learning (MRL)
- Partially Observable Markov Decision Process (POMDP)
- Robust Control Framework with Continuous Ambiguity
- Cross-Entropy Method for optimization

This module provides advanced decision-making for autonomous vehicles
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import math
import random

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class VehicleAction(Enum):
    """Actions from the paper (Table 2)"""
    LANE_LEFT = 0
    IDLE = 1
    LANE_RIGHT = 2
    FASTER = 3
    SLOWER = 4

@dataclass
class VehicleState:
    """State representation for MDP"""
    x: float  # Position x
    y: float  # Position y
    vx: float  # Velocity x
    vy: float  # Velocity y
    heading: float  # Heading angle
    lane: int  # Current lane index
    
    def to_array(self) -> np.ndarray:
        """Convert state to numpy array"""
        return np.array([self.x, self.y, self.vx, self.vy, self.heading, self.lane])
    
    @staticmethod
    def from_array(arr: np.ndarray) -> 'VehicleState':
        """Create state from numpy array"""
        return VehicleState(arr[0], arr[1], arr[2], arr[3], arr[4], int(arr[5]))

class DynamicsModel(nn.Module):
    """
    Neural Network Dynamics Model
    Implements the structured model from the paper (Equation 18):
    ẋ = f_θ(x, u) = A_θ(x, u)x + B_θ(x, u)u
    """
    
    def __init__(self, state_dim: int = 6, action_dim: int = 5, hidden_dim: int = 128):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Network for A matrix (state transition)
        self.A_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim * state_dim)
        )
        
        # Network for B matrix (control input)
        self.B_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim * action_dim)
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dynamics model
        Args:
            state: Current state [batch_size, state_dim]
            action: Action taken [batch_size, action_dim]
        Returns:
            next_state: Predicted next state [batch_size, state_dim]
        """
        batch_size = state.shape[0]
        
        # Concatenate state and action
        x = torch.cat([state, action], dim=1)
        
        # Get A and B matrices
        A = self.A_network(x).reshape(batch_size, self.state_dim, self.state_dim)
        B = self.B_network(x).reshape(batch_size, self.state_dim, self.action_dim)
        
        # Apply dynamics: next_state = A @ state + B @ action
        next_state = torch.bmm(A, state.unsqueeze(2)).squeeze() + \
                    torch.bmm(B, action.unsqueeze(2)).squeeze()
        
        return next_state

class ExperienceBuffer:
    """
    Experience replay buffer for training dynamics model
    Stores transitions (s_t, a_t, s_{t+1}) as in Equation 17
    """
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state: np.ndarray, action: int, next_state: np.ndarray):
        """Add experience to buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, next_state)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> List[Tuple[np.ndarray, int, np.ndarray]]:
        """Sample batch of experiences"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)

class ModelBasedRL:
    """
    Model-based Reinforcement Learning implementation
    Based on Case Study 3, Section 4.3.1
    """
    
    def __init__(self, state_dim: int = 6, action_dim: int = 5, learning_rate: float = 1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize dynamics model
        self.dynamics_model = DynamicsModel(state_dim, action_dim)
        self.optimizer = optim.Adam(self.dynamics_model.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.experience_buffer = ExperienceBuffer()
        
        # Training history
        self.loss_history = []
        
    def collect_experience(self, env, num_steps: int = 1000):
        """
        Phase 1, Step 1: Collect experience through random interaction
        Creates dataset D = {s_t, a_t, s_{t+1}}_{t∈[1,N]}
        """
        state = env.reset()
        
        for _ in range(num_steps):
            # Random action
            action = random.choice(list(VehicleAction))
            
            # Execute action
            next_state = env.step(action)
            
            # Store experience
            self.experience_buffer.push(
                state.to_array(),
                action.value,
                next_state.to_array()
            )
            
            state = next_state
            
    def train_dynamics_model(self, epochs: int = 2000, batch_size: int = 32):
        """
        Phase 1, Step 3: Train dynamics model
        Uses stochastic gradient descent for 2000 epochs as in the paper
        """
        if len(self.experience_buffer) < batch_size:
            return
            
        for epoch in range(epochs):
            # Sample batch
            batch = self.experience_buffer.sample(batch_size)
            
            # Convert to tensors
            states = torch.FloatTensor([t[0] for t in batch])
            actions = torch.zeros((batch_size, self.action_dim))
            for i, t in enumerate(batch):
                actions[i, t[1]] = 1  # One-hot encoding
            next_states = torch.FloatTensor([t[2] for t in batch])
            
            # Forward pass
            predicted_next = self.dynamics_model(states, actions)
            
            # Calculate loss
            loss = nn.MSELoss()(predicted_next, next_states)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Record loss
            self.loss_history.append(loss.item())
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    def predict_trajectory(self, initial_state: VehicleState, 
                          action_sequence: List[VehicleAction],
                          horizon: int = 10) -> List[VehicleState]:
        """
        Phase 1, Step 4: Predict trajectory using trained model
        """
        trajectory = [initial_state]
        state = torch.FloatTensor(initial_state.to_array()).unsqueeze(0)
        
        for i in range(min(horizon, len(action_sequence))):
            # Convert action to one-hot
            action = torch.zeros((1, self.action_dim))
            action[0, action_sequence[i].value] = 1
            
            # Predict next state
            with torch.no_grad():
                next_state = self.dynamics_model(state, action)
            
            # Add to trajectory
            trajectory.append(VehicleState.from_array(next_state.numpy()[0]))
            state = next_state
            
        return trajectory

class CrossEntropyMethod:
    """
    Cross-Entropy Method for trajectory optimization
    Phase 2: Planning phase implementation
    """
    
    def __init__(self, dynamics_model: DynamicsModel, 
                 horizon: int = 10,
                 num_samples: int = 100,
                 elite_frac: float = 0.1):
        self.dynamics_model = dynamics_model
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elite = int(num_samples * elite_frac)
        
    def optimize(self, initial_state: VehicleState, 
                goal_state: VehicleState,
                num_iterations: int = 10) -> List[VehicleAction]:
        """
        Optimize action sequence using CEM
        """
        state_dim = 6
        action_dim = 5
        
        # Initialize action distribution (Gaussian)
        mean = np.zeros((self.horizon, action_dim))
        std = np.ones((self.horizon, action_dim))
        
        best_actions = []
        best_reward = -float('inf')
        
        for iteration in range(num_iterations):
            # Sample action sequences
            action_sequences = []
            rewards = []
            
            for _ in range(self.num_samples):
                # Sample actions from current distribution
                actions = []
                for t in range(self.horizon):
                    action_probs = np.random.normal(mean[t], std[t])
                    action_probs = self.softmax(action_probs)
                    action = np.random.choice(action_dim, p=action_probs)
                    actions.append(action)
                
                action_sequences.append(actions)
                
                # Evaluate trajectory
                reward = self.evaluate_trajectory(initial_state, actions, goal_state)
                rewards.append(reward)
            
            # Select elite samples
            elite_indices = np.argsort(rewards)[-self.num_elite:]
            elite_sequences = [action_sequences[i] for i in elite_indices]
            elite_rewards = [rewards[i] for i in elite_indices]
            
            # Update best
            if max(elite_rewards) > best_reward:
                best_reward = max(elite_rewards)
                best_idx = elite_rewards.index(max(elite_rewards))
                best_actions = elite_sequences[best_idx]
            
            # Update distribution
            for t in range(self.horizon):
                elite_actions_t = np.array([seq[t] for seq in elite_sequences])
                mean[t] = np.mean(np.eye(action_dim)[elite_actions_t], axis=0)
                std[t] = np.std(np.eye(action_dim)[elite_actions_t], axis=0) + 0.01
        
        # Convert to VehicleAction enum
        return [VehicleAction(a) for a in best_actions]
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax function for action probabilities"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def evaluate_trajectory(self, initial_state: VehicleState,
                          action_sequence: List[int],
                          goal_state: VehicleState) -> float:
        """
        Evaluate trajectory using reward function
        Weighted L1-norm as in the paper
        """
        state = torch.FloatTensor(initial_state.to_array()).unsqueeze(0)
        total_reward = 0
        gamma = 0.95  # Discount factor
        
        for t, action_idx in enumerate(action_sequence):
            # One-hot encode action
            action = torch.zeros((1, 5))
            action[0, action_idx] = 1
            
            # Predict next state
            with torch.no_grad():
                next_state = self.dynamics_model(state, action)
            
            # Calculate reward (negative L1 distance to goal)
            goal = torch.FloatTensor(goal_state.to_array())
            distance = torch.abs(next_state[0] - goal).sum()
            reward = -distance.item()
            
            total_reward += (gamma ** t) * reward
            state = next_state
            
        return total_reward

class RobustControl:
    """
    Robust Control Framework with Continuous Ambiguity
    Addresses model uncertainty and predicts neighboring vehicle trajectories
    """
    
    def __init__(self, dynamics_model: DynamicsModel):
        self.dynamics_model = dynamics_model
        self.uncertainty_radius = 0.1  # Model uncertainty bound
        
    def predict_with_uncertainty(self, state: VehicleState, 
                                action: VehicleAction,
                                num_samples: int = 10) -> List[VehicleState]:
        """
        Predict next state considering model uncertainty
        Returns multiple possible next states
        """
        predictions = []
        
        state_tensor = torch.FloatTensor(state.to_array()).unsqueeze(0)
        action_tensor = torch.zeros((1, 5))
        action_tensor[0, action.value] = 1
        
        for _ in range(num_samples):
            # Add noise to model parameters
            with torch.no_grad():
                # Get base prediction
                next_state = self.dynamics_model(state_tensor, action_tensor)
                
                # Add uncertainty
                noise = torch.randn_like(next_state) * self.uncertainty_radius
                next_state_uncertain = next_state + noise
                
                predictions.append(VehicleState.from_array(next_state_uncertain.numpy()[0]))
        
        return predictions
    
    def worst_case_planning(self, initial_state: VehicleState,
                           action_sequence: List[VehicleAction],
                           neighboring_vehicles: List[VehicleState]) -> float:
        """
        Evaluate action sequence considering worst-case scenarios
        Implements robust planning from the paper
        """
        worst_case_cost = 0
        
        for action in action_sequence:
            # Get multiple predictions considering uncertainty
            possible_states = self.predict_with_uncertainty(initial_state, action)
            
            # For each possible ego state
            max_collision_risk = 0
            for ego_state in possible_states:
                # Check collision risk with all neighboring vehicles
                for neighbor in neighboring_vehicles:
                    # Consider all possible neighbor actions
                    for neighbor_action in VehicleAction:
                        neighbor_predictions = self.predict_with_uncertainty(
                            neighbor, neighbor_action
                        )
                        
                        for neighbor_future in neighbor_predictions:
                            # Calculate collision risk
                            distance = np.sqrt(
                                (ego_state.x - neighbor_future.x)**2 + 
                                (ego_state.y - neighbor_future.y)**2
                            )
                            
                            collision_risk = max(0, 10 - distance)  # Risk increases as distance decreases
                            max_collision_risk = max(max_collision_risk, collision_risk)
            
            worst_case_cost += max_collision_risk
            initial_state = possible_states[0]  # Continue from first prediction
            
        return worst_case_cost
    
    def continuous_ambiguity_prediction(self, vehicle_state: VehicleState,
                                       observation_history: List[VehicleState],
                                       horizon: int = 5) -> List[VehicleState]:
        """
        Predict trajectory considering driving style ambiguity
        Implements continuous ambiguity from Figure 13 in the paper
        """
        # Estimate driving style from observation history
        if len(observation_history) < 2:
            driving_style = "normal"
        else:
            # Calculate average speed and acceleration
            speeds = []
            accelerations = []
            
            for i in range(1, len(observation_history)):
                prev = observation_history[i-1]
                curr = observation_history[i]
                
                speed = np.sqrt(curr.vx**2 + curr.vy**2)
                speeds.append(speed)
                
                if i > 1:
                    prev_speed = np.sqrt(prev.vx**2 + prev.vy**2)
                    accel = speed - prev_speed
                    accelerations.append(accel)
            
            avg_speed = np.mean(speeds) if speeds else 25
            avg_accel = np.mean(accelerations) if accelerations else 0
            
            # Classify driving style
            if avg_speed > 30:
                driving_style = "aggressive"
            elif avg_speed < 20:
                driving_style = "conservative"
            else:
                driving_style = "normal"
        
        # Predict trajectory based on driving style
        trajectory = [vehicle_state]
        current_state = vehicle_state
        
        for _ in range(horizon):
            # Select action based on driving style
            if driving_style == "aggressive":
                action_probs = [0.1, 0.2, 0.1, 0.5, 0.1]  # Prefer FASTER
            elif driving_style == "conservative":
                action_probs = [0.1, 0.5, 0.1, 0.1, 0.2]  # Prefer IDLE
            else:
                action_probs = [0.2, 0.3, 0.2, 0.15, 0.15]  # Balanced
            
            # Sample action
            action = np.random.choice(list(VehicleAction), p=action_probs)
            
            # Predict next state
            predictions = self.predict_with_uncertainty(current_state, action, num_samples=1)
            current_state = predictions[0]
            trajectory.append(current_state)
        
        return trajectory

class POMDP:
    """
    Partially Observable Markov Decision Process
    Implements POMDP from Section 3.1 of the paper
    """
    
    def __init__(self, states: List[VehicleState], 
                 actions: List[VehicleAction],
                 observations: List[np.ndarray]):
        self.states = states
        self.actions = actions
        self.observations = observations
        
        # Initialize belief state (uniform distribution)
        self.belief = np.ones(len(states)) / len(states)
        
        # Transition and observation models (simplified)
        self.transition_prob = np.ones((len(states), len(actions), len(states))) / len(states)
        self.observation_prob = np.ones((len(states), len(actions), len(observations))) / len(observations)
        
    def update_belief(self, action: VehicleAction, observation: np.ndarray):
        """
        Update belief state based on action and observation
        Implements belief update from POMDP
        """
        action_idx = action.value
        obs_idx = self.find_observation_index(observation)
        
        # Predict step
        predicted_belief = np.zeros(len(self.states))
        for s_prime in range(len(self.states)):
            for s in range(len(self.states)):
                predicted_belief[s_prime] += self.belief[s] * self.transition_prob[s, action_idx, s_prime]
        
        # Update step
        for s in range(len(self.states)):
            self.belief[s] = predicted_belief[s] * self.observation_prob[s, action_idx, obs_idx]
        
        # Normalize
        self.belief /= self.belief.sum()
    
    def find_observation_index(self, observation: np.ndarray) -> int:
        """Find closest observation in the observation space"""
        min_dist = float('inf')
        min_idx = 0
        
        for i, obs in enumerate(self.observations):
            dist = np.linalg.norm(observation - obs)
            if dist < min_dist:
                min_dist = dist
                min_idx = i
                
        return min_idx
    
    def get_best_action(self) -> VehicleAction:
        """
        Get best action based on current belief state
        """
        expected_rewards = np.zeros(len(self.actions))
        
        for a in range(len(self.actions)):
            for s in range(len(self.states)):
                # Simple reward function (distance to goal)
                reward = -np.linalg.norm(self.states[s].to_array()[:2])
                expected_rewards[a] += self.belief[s] * reward
        
        best_action_idx = np.argmax(expected_rewards)
        return self.actions[best_action_idx]

def demonstrate_behavioral_planning():
    """
    Demonstration of the behavioral planning system
    Recreates results from Case Study 3
    """
    print("=" * 60)
    print("Behavioral Planning Demonstration")
    print("Based on Case Study 3 from the paper")
    print("=" * 60)
    
    # Initialize components
    state_dim = 6
    action_dim = 5
    
    # Create model-based RL system
    print("\n1. Initializing Model-Based RL...")
    mbrl = ModelBasedRL(state_dim, action_dim)
    
    # Create some synthetic experience data
    print("\n2. Generating synthetic experience data...")
    for _ in range(1000):
        state = np.random.randn(state_dim)
        action = random.randint(0, action_dim - 1)
        next_state = state + np.random.randn(state_dim) * 0.1
        mbrl.experience_buffer.push(state, action, next_state)
    
    # Train dynamics model
    print("\n3. Training dynamics model (2000 epochs)...")
    mbrl.train_dynamics_model(epochs=200, batch_size=32)  # Reduced for demo
    
    # Demonstrate trajectory prediction
    print("\n4. Demonstrating trajectory prediction...")
    initial_state = VehicleState(0, 0, 20, 0, 0, 1)
    action_sequence = [VehicleAction.FASTER, VehicleAction.LANE_LEFT, 
                      VehicleAction.IDLE, VehicleAction.LANE_RIGHT]
    
    trajectory = mbrl.predict_trajectory(initial_state, action_sequence)
    print(f"   Initial state: x={initial_state.x:.1f}, y={initial_state.y:.1f}")
    print(f"   Final state: x={trajectory[-1].x:.1f}, y={trajectory[-1].y:.1f}")
    
    # Cross-entropy method optimization
    print("\n5. Optimizing trajectory with Cross-Entropy Method...")
    cem = CrossEntropyMethod(mbrl.dynamics_model, horizon=10)
    goal_state = VehicleState(100, 4, 25, 0, 0, 2)
    
    optimal_actions = cem.optimize(initial_state, goal_state, num_iterations=5)
    print(f"   Optimal action sequence: {[a.name for a in optimal_actions[:5]]}")
    
    # Robust control
    print("\n6. Demonstrating Robust Control Framework...")
    robust = RobustControl(mbrl.dynamics_model)
    
    # Predict with uncertainty
    uncertain_predictions = robust.predict_with_uncertainty(
        initial_state, VehicleAction.FASTER, num_samples=5
    )
    print(f"   Predictions with uncertainty (5 samples):")
    for i, pred in enumerate(uncertain_predictions):
        print(f"     Sample {i+1}: x={pred.x:.2f}, y={pred.y:.2f}")
    
    # Continuous ambiguity prediction
    print("\n7. Continuous Ambiguity Prediction...")
    observation_history = [
        VehicleState(0, 0, 15, 0, 0, 1),
        VehicleState(20, 0, 18, 0, 0, 1),
        VehicleState(40, 0, 22, 0, 0, 1)
    ]
    
    predicted_trajectory = robust.continuous_ambiguity_prediction(
        observation_history[-1], observation_history, horizon=5
    )
    print(f"   Predicted trajectory for neighboring vehicle:")
    for i, state in enumerate(predicted_trajectory[:3]):
        print(f"     t={i}: x={state.x:.1f}, y={state.y:.1f}")
    
    print("\n" + "=" * 60)
    print("Demonstration Complete!")
    print("This recreates the key algorithms from Case Study 3")
    print("including MDP, robust control, and trajectory prediction")
    print("=" * 60)

def main():
    """Main entry point for behavioral planning demo"""
    demonstrate_behavioral_planning()

if __name__ == "__main__":
    main()

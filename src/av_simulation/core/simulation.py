"""
Autonomous Vehicle Simulation Environment
Based on the paper: "Safety and Risk Analysis of Autonomous Vehicles Using Computer Vision and Neural Networks"

This simulation includes three environments:
1. Highway environment - 4-lane highway navigation
2. Lane merging environment - Highway with service road merge
3. Roundabout environment - Roundabout navigation with 4 incoming roads

Simulation parameters from the paper:
- Acceleration Range: (-5, 5.0) m/s²
- Steering Range: (-0.785, 0.785) rad (~45 degrees)
- Max Speed: 40 m/s
- Default Speeds: [23, 25] m/s
- Perception Distance: 180 m
"""

import pygame
import math
import random
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional
import sys

# Import data collection system
try:
    from ..data.repository import DataRepository
    from ..data.collectors import DataCollectionManager
    from ..data.exporters import CSVExporter, JSONExporter
    DATA_COLLECTION_AVAILABLE = True
except ImportError:
    DATA_COLLECTION_AVAILABLE = False
    print("Warning: Data collection system not available")

# Initialize Pygame
pygame.init()

# Constants from the paper
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)
GREEN = (0, 255, 0)  # Ego vehicle color
BLUE = (0, 100, 255)  # Other vehicles
RED = (255, 0, 0)    # Collision/danger
YELLOW = (255, 255, 0)  # Lane markings
LIGHT_GRAY = (200, 200, 200)

# Simulation Parameters (from Table 2 in the paper)
ACCELERATION_RANGE = (-5.0, 5.0)  # m/s²
STEERING_RANGE = (-0.7853981633974483, 0.7853981633974483)  # rad
MAX_SPEED = 40.0  # m/s
DEFAULT_SPEEDS = [23.0, 25.0]  # m/s
DISTANCE_WANTED = 10.0  # m
TIME_WANTED = 1.5  # s
STRIPE_SPACING = 5  # m
STRIPE_LENGTH = 3  # m
STRIPE_WIDTH = 0.3  # m
PERCEPTION_DISTANCE = 180  # m
PIXELS_PER_METER = 2  # Scale factor

class Action(Enum):
    """Actions as defined in the paper"""
    LANE_LEFT = 0
    IDLE = 1
    LANE_RIGHT = 2
    FASTER = 3
    SLOWER = 4
    # Aliases for test compatibility
    ACCELERATE = 3    # Alias for FASTER
    DECELERATE = 4    # Alias for SLOWER
    BRAKE = 4         # Alias for SLOWER
    TURN_LEFT = 0     # Alias for LANE_LEFT
    TURN_RIGHT = 2    # Alias for LANE_RIGHT
    MAINTAIN = 1      # Alias for IDLE

@dataclass
class VehicleState:
    """Vehicle state representation"""
    x: float  # Position x
    y: float  # Position y
    vx: float  # Velocity x
    vy: float  # Velocity y
    heading: float  # Heading angle in radians
    acceleration: float  # Current acceleration
    steering_angle: float  # Current steering angle

    # Constructor that supports both old and new style initialization
    def __init__(self, x: float, y: float, vx: float = None, vy: float = None,
                 heading: float = None, acceleration: float = 0, steering_angle: float = 0,
                 # Legacy parameters for test compatibility
                 angle: float = None, speed: float = None):
        self.x = x
        self.y = y

        # Handle legacy angle/speed parameters
        if angle is not None:
            heading = angle
        if speed is not None:
            vx = speed if heading is not None else speed
            vy = 0

        self.vx = vx if vx is not None else 0
        self.vy = vy if vy is not None else 0
        self.heading = heading if heading is not None else 0
        self.acceleration = acceleration
        self.steering_angle = steering_angle

    @property
    def angle(self):
        """Alias for heading for test compatibility"""
        return self.heading

    @property
    def speed(self):
        """Calculate speed from velocity components"""
        return math.sqrt(self.vx**2 + self.vy**2)
    
class Vehicle:
    """Base vehicle class with dynamics model"""
    
    def __init__(self, x: float, y: float, vx: float = 0, vy: float = 0,
                 heading: float = 0, color: Tuple[int, int, int] = BLUE,
                 is_ego: bool = False,
                 # Legacy parameters for test compatibility
                 angle: float = None, speed: float = None):
        # Handle legacy parameters
        if angle is not None:
            heading = angle
        if speed is not None:
            vx = speed
            vy = 0

        self.state = VehicleState(x, y, vx, vy, heading, 0, 0)
        self.color = color
        self.is_ego = is_ego
        self.width = 4.0  # meters
        self.length = 5.0  # meters
        self.max_acceleration = ACCELERATION_RANGE[1]
        self.min_acceleration = ACCELERATION_RANGE[0]
        self.max_steering = STEERING_RANGE[1]
        self.min_steering = STEERING_RANGE[0]

        # Additional properties for test compatibility
        self._target_speed = speed if speed is not None else DEFAULT_SPEEDS[0]
        self.is_autonomous = is_ego  # For test compatibility

    # Add properties for test compatibility
    @property
    def x(self):
        return self.state.x

    @x.setter
    def x(self, value):
        self.state.x = value

    @property
    def y(self):
        return self.state.y

    @y.setter
    def y(self, value):
        self.state.y = value

    @property
    def angle(self):
        return self.state.heading

    @property
    def speed(self):
        return math.sqrt(self.state.vx**2 + self.state.vy**2)

    @property
    def target_speed(self):
        return self._target_speed

    @target_speed.setter
    def target_speed(self, value):
        self._target_speed = value

    def distance_to(self, other_vehicle):
        """Calculate distance to another vehicle"""
        return math.sqrt((self.state.x - other_vehicle.state.x)**2 +
                        (self.state.y - other_vehicle.state.y)**2)

    def adjust_speed(self, new_speed):
        """Adjust vehicle speed while maintaining direction"""
        current_speed = self.speed
        if current_speed > 0:
            ratio = new_speed / current_speed
            self.state.vx *= ratio
            self.state.vy *= ratio
        else:
            self.state.vx = new_speed
            self.state.vy = 0

    def is_colliding_with(self, other_vehicle):
        """Check collision with another vehicle"""
        return self.get_rect().colliderect(other_vehicle.get_rect())
        
    def update(self, dt: float):
        """Update vehicle position using dynamics model from the paper"""
        # Update velocity
        speed = math.sqrt(self.state.vx**2 + self.state.vy**2)
        new_speed = speed + self.state.acceleration * dt
        new_speed = max(0, min(new_speed, MAX_SPEED))
        
        # Update heading based on steering angle
        if speed > 0:
            self.state.heading += (self.state.steering_angle * speed / self.length) * dt
        
        # Update velocity components
        self.state.vx = new_speed * math.cos(self.state.heading)
        self.state.vy = new_speed * math.sin(self.state.heading)
        
        # Update position
        self.state.x += self.state.vx * dt
        self.state.y += self.state.vy * dt
        
    def set_action(self, action: Action):
        """Apply action to vehicle"""
        if action == Action.FASTER:
            self.state.acceleration = 2.0
        elif action == Action.SLOWER:
            self.state.acceleration = -2.0
        elif action == Action.IDLE:
            self.state.acceleration = 0
        # Lane changes handled by environment
            
    def get_rect(self) -> pygame.Rect:
        """Get vehicle rectangle for collision detection"""
        return pygame.Rect(
            (self.state.x - self.length/2) * PIXELS_PER_METER,
            (self.state.y - self.width/2) * PIXELS_PER_METER,
            self.length * PIXELS_PER_METER,
            self.width * PIXELS_PER_METER
        )
        
    def draw(self, screen: pygame.Surface, offset_x: int = 0, offset_y: int = 0):
        """Draw vehicle on screen"""
        # Create vehicle surface
        vehicle_surface = pygame.Surface((self.length * PIXELS_PER_METER, 
                                         self.width * PIXELS_PER_METER))
        vehicle_surface.fill(self.color)
        
        # Add windshield indicator
        if self.is_ego:
            pygame.draw.rect(vehicle_surface, WHITE, 
                           (int(self.length * PIXELS_PER_METER * 0.7), 0,
                            int(self.length * PIXELS_PER_METER * 0.3), 
                            self.width * PIXELS_PER_METER))
        
        # Rotate vehicle based on heading
        rotated = pygame.transform.rotate(vehicle_surface, -math.degrees(self.state.heading))
        
        # Position on screen
        rect = rotated.get_rect()
        rect.center = (int(self.state.x * PIXELS_PER_METER) + offset_x,
                      int(self.state.y * PIXELS_PER_METER) + offset_y)
        screen.blit(rotated, rect)

class Environment:
    """Base environment class"""

    def __init__(self, width: int = SCREEN_WIDTH, height: int = SCREEN_HEIGHT):
        self.vehicles = []
        self.ego_vehicle = None
        self.time = 0
        self.collision_detected = False
        self.width = width
        self.height = height
        self.lanes = []  # For test compatibility
        
    def reset(self):
        """Reset environment to initial state"""
        raise NotImplementedError
        
    def step(self, dt: float):
        """Update environment by one timestep"""
        self.time += dt
        
        # Update all vehicles
        for vehicle in self.vehicles:
            vehicle.update(dt)
            
        # Check collisions
        self.check_collisions()
        
    def check_collisions(self):
        """Check for collisions between ego vehicle and others"""
        if not self.ego_vehicle:
            return
            
        ego_rect = self.ego_vehicle.get_rect()
        for vehicle in self.vehicles:
            if vehicle != self.ego_vehicle:
                if ego_rect.colliderect(vehicle.get_rect()):
                    self.collision_detected = True
                    return
                    
    def draw(self, screen: pygame.Surface):
        """Draw environment"""
        raise NotImplementedError

class HighwayEnvironment(Environment):
    """Highway environment with 4 lanes"""

    def __init__(self, width: int = SCREEN_WIDTH, height: int = SCREEN_HEIGHT):
        super().__init__(width, height)
        self.num_lanes = 4
        self.lane_width = 4.0  # meters
        self.road_length = 1000  # meters
        self.reset()
        
    def reset(self):
        """Initialize highway environment"""
        self.vehicles = []
        self.collision_detected = False
        
        # Create ego vehicle in lane 2
        self.ego_vehicle = Vehicle(
            x=50, 
            y=self.get_lane_center(2),
            vx=DEFAULT_SPEEDS[0],
            color=GREEN,
            is_ego=True
        )
        self.vehicles.append(self.ego_vehicle)
        
        # Create other vehicles
        for i in range(8):
            lane = random.randint(0, self.num_lanes - 1)
            x = random.uniform(100, 400)
            speed = random.uniform(DEFAULT_SPEEDS[0] - 5, DEFAULT_SPEEDS[1] + 5)
            
            vehicle = Vehicle(
                x=x,
                y=self.get_lane_center(lane),
                vx=speed,
                color=BLUE
            )
            self.vehicles.append(vehicle)
            
    def get_lane_center(self, lane: int) -> float:
        """Get y-coordinate of lane center"""
        return SCREEN_HEIGHT / (2 * PIXELS_PER_METER) + (lane - 1.5) * self.lane_width
        
    def draw(self, screen: pygame.Surface):
        """Draw highway environment"""
        screen.fill(DARK_GRAY)
        
        # Calculate camera offset to follow ego vehicle
        if self.ego_vehicle:
            camera_x = int(SCREEN_WIDTH/2 - self.ego_vehicle.state.x * PIXELS_PER_METER)
        else:
            camera_x = 0
            
        # Draw road
        road_top = SCREEN_HEIGHT // 2 - int(2 * self.lane_width * PIXELS_PER_METER)
        road_height = int(self.num_lanes * self.lane_width * PIXELS_PER_METER)
        pygame.draw.rect(screen, GRAY, (0, road_top, SCREEN_WIDTH, road_height))
        
        # Draw lane markings
        for i in range(1, self.num_lanes):
            y = road_top + int(i * self.lane_width * PIXELS_PER_METER)
            
            # Draw dashed line
            for x in range(0, SCREEN_WIDTH, int((STRIPE_LENGTH + STRIPE_SPACING) * PIXELS_PER_METER)):
                pygame.draw.rect(screen, WHITE,
                               (x, y - int(STRIPE_WIDTH * PIXELS_PER_METER / 2),
                                int(STRIPE_LENGTH * PIXELS_PER_METER),
                                int(STRIPE_WIDTH * PIXELS_PER_METER)))
        
        # Draw vehicles
        for vehicle in self.vehicles:
            vehicle.draw(screen, camera_x, 0)
            
        # Draw collision indicator
        if self.collision_detected:
            font = pygame.font.Font(None, 72)
            text = font.render("COLLISION!", True, RED)
            screen.blit(text, (SCREEN_WIDTH // 2 - 150, 50))

class MergingEnvironment(Environment):
    """Lane merging environment"""

    def __init__(self, width: int = SCREEN_WIDTH, height: int = SCREEN_HEIGHT):
        super().__init__(width, height)
        self.num_lanes = 4
        self.lane_width = 4.0
        self.merge_point = 300  # meters
        self.reset()
        
    def reset(self):
        """Initialize merging environment"""
        self.vehicles = []
        self.collision_detected = False
        
        # Create ego vehicle
        self.ego_vehicle = Vehicle(
            x=50,
            y=self.get_lane_center(2),
            vx=DEFAULT_SPEEDS[0],
            color=GREEN,
            is_ego=True
        )
        self.vehicles.append(self.ego_vehicle)
        
        # Create merging vehicle
        merge_vehicle = Vehicle(
            x=self.merge_point - 50,
            y=self.get_lane_center(0) - self.lane_width * 2,  # Service lane
            vx=DEFAULT_SPEEDS[0] - 5,
            vy=2,  # Moving toward main road
            heading=0.1,  # Slight angle
            color=YELLOW
        )
        self.vehicles.append(merge_vehicle)
        
        # Create other highway vehicles
        for i in range(5):
            lane = random.randint(0, self.num_lanes - 1)
            x = random.uniform(100, 400)
            speed = random.uniform(DEFAULT_SPEEDS[0] - 5, DEFAULT_SPEEDS[1])
            
            vehicle = Vehicle(
                x=x,
                y=self.get_lane_center(lane),
                vx=speed,
                color=BLUE
            )
            self.vehicles.append(vehicle)
            
    def get_lane_center(self, lane: int) -> float:
        """Get y-coordinate of lane center"""
        return SCREEN_HEIGHT / (2 * PIXELS_PER_METER) + (lane - 1.5) * self.lane_width
        
    def draw(self, screen: pygame.Surface):
        """Draw merging environment"""
        screen.fill(DARK_GRAY)
        
        # Camera follow ego vehicle
        if self.ego_vehicle:
            camera_x = int(SCREEN_WIDTH/2 - self.ego_vehicle.state.x * PIXELS_PER_METER)
        else:
            camera_x = 0
            
        # Draw main road
        road_top = SCREEN_HEIGHT // 2 - int(2 * self.lane_width * PIXELS_PER_METER)
        road_height = int(self.num_lanes * self.lane_width * PIXELS_PER_METER)
        pygame.draw.rect(screen, GRAY, (0, road_top, SCREEN_WIDTH, road_height))
        
        # Draw service/merge lane
        merge_lane_y = road_top - int(self.lane_width * PIXELS_PER_METER * 1.5)
        merge_start_x = int((self.merge_point - 100) * PIXELS_PER_METER) + camera_x
        merge_width = int(150 * PIXELS_PER_METER)
        
        # Draw merge lane road
        pygame.draw.rect(screen, GRAY, 
                        (merge_start_x, merge_lane_y, 
                         merge_width, int(self.lane_width * PIXELS_PER_METER)))
        
        # Draw merge connection
        pygame.draw.polygon(screen, GRAY, [
            (merge_start_x + merge_width, merge_lane_y),
            (merge_start_x + merge_width + 50, road_top),
            (merge_start_x + merge_width + 50, road_top + int(self.lane_width * PIXELS_PER_METER)),
            (merge_start_x + merge_width, merge_lane_y + int(self.lane_width * PIXELS_PER_METER))
        ])
        
        # Draw lane markings
        for i in range(1, self.num_lanes):
            y = road_top + int(i * self.lane_width * PIXELS_PER_METER)
            for x in range(0, SCREEN_WIDTH, int((STRIPE_LENGTH + STRIPE_SPACING) * PIXELS_PER_METER)):
                pygame.draw.rect(screen, WHITE,
                               (x, y - int(STRIPE_WIDTH * PIXELS_PER_METER / 2),
                                int(STRIPE_LENGTH * PIXELS_PER_METER),
                                int(STRIPE_WIDTH * PIXELS_PER_METER)))
        
        # Draw vehicles
        for vehicle in self.vehicles:
            vehicle.draw(screen, camera_x, 0)
            
        # Draw merge point indicator
        merge_x = int(self.merge_point * PIXELS_PER_METER) + camera_x
        pygame.draw.line(screen, YELLOW, (merge_x, road_top - 20), (merge_x, road_top + road_height + 20), 3)
        
        if self.collision_detected:
            font = pygame.font.Font(None, 72)
            text = font.render("COLLISION!", True, RED)
            screen.blit(text, (SCREEN_WIDTH // 2 - 150, 50))

class RoundaboutEnvironment(Environment):
    """Roundabout environment with 4 entrances"""

    def __init__(self, width: int = SCREEN_WIDTH, height: int = SCREEN_HEIGHT):
        super().__init__(width, height)
        self.center_x = width / (2 * PIXELS_PER_METER)
        self.center_y = height / (2 * PIXELS_PER_METER)
        self.outer_radius = 80  # meters
        self.inner_radius = 40  # meters
        self.reset()
        
    def reset(self):
        """Initialize roundabout environment"""
        self.vehicles = []
        self.collision_detected = False
        
        # Create ego vehicle approaching from bottom
        self.ego_vehicle = Vehicle(
            x=self.center_x,
            y=self.center_y + self.outer_radius + 30,
            vx=0,
            vy=-DEFAULT_SPEEDS[0]/2,
            heading=-math.pi/2,
            color=GREEN,
            is_ego=True
        )
        self.vehicles.append(self.ego_vehicle)
        
        # Create vehicles in roundabout
        for i in range(4):
            angle = i * math.pi / 2
            radius = (self.outer_radius + self.inner_radius) / 2
            
            vehicle = Vehicle(
                x=self.center_x + radius * math.cos(angle),
                y=self.center_y + radius * math.sin(angle),
                vx=-10 * math.sin(angle),
                vy=10 * math.cos(angle),
                heading=angle + math.pi/2,
                color=BLUE
            )
            self.vehicles.append(vehicle)
            
    def draw(self, screen: pygame.Surface):
        """Draw roundabout environment"""
        screen.fill(LIGHT_GRAY)
        
        center_x_px = int(self.center_x * PIXELS_PER_METER)
        center_y_px = int(self.center_y * PIXELS_PER_METER)
        
        # Draw roundabout road
        pygame.draw.circle(screen, GRAY, (center_x_px, center_y_px),
                         int(self.outer_radius * PIXELS_PER_METER))
        pygame.draw.circle(screen, LIGHT_GRAY, (center_x_px, center_y_px),
                         int(self.inner_radius * PIXELS_PER_METER))
        
        # Draw approach roads
        road_width = int(20 * PIXELS_PER_METER)
        
        # Bottom road
        pygame.draw.rect(screen, GRAY,
                        (center_x_px - road_width//2,
                         center_y_px + int(self.inner_radius * PIXELS_PER_METER),
                         road_width,
                         SCREEN_HEIGHT))
        
        # Top road  
        pygame.draw.rect(screen, GRAY,
                        (center_x_px - road_width//2,
                         0,
                         road_width,
                         center_y_px - int(self.inner_radius * PIXELS_PER_METER)))
        
        # Left road
        pygame.draw.rect(screen, GRAY,
                        (0,
                         center_y_px - road_width//2,
                         center_x_px - int(self.inner_radius * PIXELS_PER_METER),
                         road_width))
        
        # Right road
        pygame.draw.rect(screen, GRAY,
                        (center_x_px + int(self.inner_radius * PIXELS_PER_METER),
                         center_y_px - road_width//2,
                         SCREEN_WIDTH,
                         road_width))
        
        # Draw lane markings in roundabout
        mid_radius = (self.outer_radius + self.inner_radius) / 2
        pygame.draw.circle(screen, WHITE, (center_x_px, center_y_px),
                         int(mid_radius * PIXELS_PER_METER), 2)
        
        # Draw vehicles
        for vehicle in self.vehicles:
            vehicle.draw(screen)
            
        if self.collision_detected:
            font = pygame.font.Font(None, 72)
            text = font.render("COLLISION!", True, RED)
            screen.blit(text, (SCREEN_WIDTH // 2 - 150, 50))

class BehaviorPlanner:
    """Implements behavioral planning using simplified MDP approach from paper"""
    
    def __init__(self, environment: Environment = None):
        self.environment = environment
        self.current_action = Action.IDLE
        
    def plan_action(self, ego_vehicle: Vehicle) -> Action:
        """Plan next action based on current state"""
        if not ego_vehicle:
            return Action.IDLE
            
        # Get nearby vehicles
        nearby_vehicles = self.get_nearby_vehicles(ego_vehicle)
        
        # Simple collision avoidance
        front_vehicle = self.get_front_vehicle(ego_vehicle, nearby_vehicles)
        
        if front_vehicle:
            distance = math.sqrt((front_vehicle.state.x - ego_vehicle.state.x)**2 + 
                               (front_vehicle.state.y - ego_vehicle.state.y)**2)
            
            if distance < DISTANCE_WANTED * 2:
                # Too close - brake or change lane
                if self.is_lane_change_safe(ego_vehicle, nearby_vehicles, Action.LANE_LEFT):
                    return Action.LANE_LEFT
                elif self.is_lane_change_safe(ego_vehicle, nearby_vehicles, Action.LANE_RIGHT):
                    return Action.LANE_RIGHT
                else:
                    return Action.SLOWER
            elif distance < DISTANCE_WANTED * 3:
                # Maintain distance
                return Action.IDLE
        
        # Default - maintain speed
        speed = math.sqrt(ego_vehicle.state.vx**2 + ego_vehicle.state.vy**2)
        if speed < DEFAULT_SPEEDS[0]:
            return Action.FASTER
        elif speed > DEFAULT_SPEEDS[1]:
            return Action.SLOWER
            
        return Action.IDLE
        
    def get_nearby_vehicles(self, ego_vehicle: Vehicle) -> List[Vehicle]:
        """Get vehicles within perception distance"""
        if not self.environment:
            return []
        nearby = []
        for vehicle in self.environment.vehicles:
            if vehicle != ego_vehicle:
                distance = math.sqrt((vehicle.state.x - ego_vehicle.state.x)**2 +
                                   (vehicle.state.y - ego_vehicle.state.y)**2)
                if distance < PERCEPTION_DISTANCE:
                    nearby.append(vehicle)
        return nearby
        
    def get_front_vehicle(self, ego_vehicle: Vehicle, nearby: List[Vehicle]) -> Optional[Vehicle]:
        """Get vehicle directly in front"""
        front_vehicle = None
        min_distance = float('inf')
        
        for vehicle in nearby:
            # Check if vehicle is in front
            if vehicle.state.x > ego_vehicle.state.x:
                # Check if in same lane (approximately)
                if abs(vehicle.state.y - ego_vehicle.state.y) < 2:
                    distance = vehicle.state.x - ego_vehicle.state.x
                    if distance < min_distance:
                        min_distance = distance
                        front_vehicle = vehicle
                        
        return front_vehicle
        
    def is_lane_change_safe(self, ego_vehicle: Vehicle, nearby: List[Vehicle], 
                           action: Action) -> bool:
        """Check if lane change is safe"""
        target_y = ego_vehicle.state.y
        if action == Action.LANE_LEFT:
            target_y -= 4  # Lane width
        elif action == Action.LANE_RIGHT:
            target_y += 4
        else:
            return True
            
        # Check for vehicles in target lane
        for vehicle in nearby:
            if abs(vehicle.state.y - target_y) < 2:
                x_distance = abs(vehicle.state.x - ego_vehicle.state.x)
                if x_distance < DISTANCE_WANTED * 2:
                    return False
                    
        return True

class Simulation:
    """Main simulation controller"""
    
    def __init__(self, enable_data_collection: bool = True):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Autonomous Vehicle Simulation - Paper Recreation")
        self.clock = pygame.time.Clock()

        # Create environments
        self.environments = {
            'highway': HighwayEnvironment(),
            'merge': MergingEnvironment(),
            'roundabout': RoundaboutEnvironment()
        }
        self.current_env_name = 'highway'
        self.current_env = self.environments[self.current_env_name]

        # Create behavior planner
        self.planner = BehaviorPlanner(self.current_env)

        # Initialize data collection system
        self.data_collection_enabled = enable_data_collection and DATA_COLLECTION_AVAILABLE
        if self.data_collection_enabled:
            self.data_repository = DataRepository("simulation_data")
            self.data_manager = DataCollectionManager(self.data_repository)
            self.current_run_id = None
            print("Data collection system initialized")
        else:
            self.data_repository = None
            self.data_manager = None

        self.running = True
        self.paused = False
        self.simulation_start_time = 0

    # Add properties for test compatibility
    @property
    def environment(self):
        return self.current_env

    @property
    def player_vehicle(self):
        return self.current_env.ego_vehicle if self.current_env else None

    def reset(self):
        """Reset current environment"""
        if self.current_env:
            self.current_env.reset()
            self.planner = BehaviorPlanner(self.current_env)
        
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.current_env.reset()
                elif event.key == pygame.K_1:
                    self.switch_environment('highway')
                elif event.key == pygame.K_2:
                    self.switch_environment('merge')
                elif event.key == pygame.K_3:
                    self.switch_environment('roundabout')
                elif event.key == pygame.K_s:
                    self.save_current_run()
                elif event.key == pygame.K_e:
                    self.export_data("csv")
                elif event.key == pygame.K_j:
                    self.export_data("json")

                # Manual control for testing
                if self.current_env.ego_vehicle:
                    if event.key == pygame.K_UP:
                        self.current_env.ego_vehicle.state.acceleration = 2
                    elif event.key == pygame.K_DOWN:
                        self.current_env.ego_vehicle.state.acceleration = -2
                    elif event.key == pygame.K_LEFT:
                        self.current_env.ego_vehicle.state.steering_angle = -0.3
                    elif event.key == pygame.K_RIGHT:
                        self.current_env.ego_vehicle.state.steering_angle = 0.3
                        
            elif event.type == pygame.KEYUP:
                if self.current_env.ego_vehicle:
                    if event.key in [pygame.K_UP, pygame.K_DOWN]:
                        self.current_env.ego_vehicle.state.acceleration = 0
                    elif event.key in [pygame.K_LEFT, pygame.K_RIGHT]:
                        self.current_env.ego_vehicle.state.steering_angle = 0
                        
    def switch_environment(self, env_name: str):
        """Switch to different environment"""
        if env_name in self.environments:
            # End current data collection run
            if self.data_collection_enabled and self.current_run_id:
                self.data_repository.end_current_run()

            self.current_env_name = env_name
            self.current_env = self.environments[env_name]
            self.current_env.reset()
            self.planner = BehaviorPlanner(self.current_env)

            # Start new data collection run
            if self.data_collection_enabled:
                self.start_data_collection()
            
    def draw_hud(self):
        """Draw heads-up display with information"""
        font = pygame.font.Font(None, 36)
        small_font = pygame.font.Font(None, 24)
        
        # Environment name
        text = font.render(f"Environment: {self.current_env_name.upper()}", True, WHITE)
        self.screen.blit(text, (10, 10))
        
        # Controls
        controls = [
            "Controls:",
            "1/2/3 - Switch Environment",
            "SPACE - Pause",
            "R - Reset",
            "Arrow Keys - Manual Control",
            "ESC - Exit"
        ]
        
        y = 60
        for control in controls:
            text = small_font.render(control, True, WHITE)
            self.screen.blit(text, (10, y))
            y += 25
            
        # Vehicle info
        if self.current_env.ego_vehicle:
            ego = self.current_env.ego_vehicle
            speed = math.sqrt(ego.state.vx**2 + ego.state.vy**2)
            
            info = [
                f"Speed: {speed:.1f} m/s",
                f"Position: ({ego.state.x:.1f}, {ego.state.y:.1f})",
                f"Heading: {math.degrees(ego.state.heading):.1f}°"
            ]
            
            y = 250
            for line in info:
                text = small_font.render(line, True, WHITE)
                self.screen.blit(text, (10, y))
                y += 25
                
        # Data collection status
        if self.data_collection_enabled:
            status_text = f"Data Collection: ON (Run: {self.current_run_id[:8] if self.current_run_id else 'None'})"
            text = small_font.render(status_text, True, GREEN)
            self.screen.blit(text, (10, SCREEN_HEIGHT - 50))

            # Show export controls
            export_text = "S - Save Data, E - Export CSV, J - Export JSON"
            text = small_font.render(export_text, True, WHITE)
            self.screen.blit(text, (10, SCREEN_HEIGHT - 25))

        # Simulation status
        if self.paused:
            pause_text = font.render("PAUSED", True, YELLOW)
            rect = pause_text.get_rect(center=(SCREEN_WIDTH // 2, 50))
            self.screen.blit(pause_text, rect)

    def start_data_collection(self):
        """Start data collection for current environment"""
        if self.data_collection_enabled:
            metadata = {
                "environment_type": self.current_env_name,
                "screen_width": SCREEN_WIDTH,
                "screen_height": SCREEN_HEIGHT,
                "fps": FPS,
                "pixels_per_meter": PIXELS_PER_METER
            }
            self.current_run_id = self.data_repository.start_new_run(
                self.current_env_name, metadata
            )
            self.simulation_start_time = pygame.time.get_ticks() / 1000.0

    def collect_simulation_data(self):
        """Collect data during simulation"""
        if self.data_collection_enabled and self.current_run_id:
            current_time = pygame.time.get_ticks() / 1000.0 - self.simulation_start_time
            self.data_manager.collect_all_data(self.current_env, current_time)

    def save_current_run(self):
        """Save current simulation run"""
        if self.data_collection_enabled and self.current_run_id:
            self.data_repository.end_current_run()
            print(f"Simulation run {self.current_run_id} saved")
            self.current_run_id = None

    def export_data(self, format_type: str = "csv"):
        """Export simulation data"""
        if not self.data_collection_enabled:
            print("Data collection not enabled")
            return

        try:
            if format_type.lower() == "csv":
                exporter = CSVExporter(self.data_repository)
                exporter.export_all_runs("exported_data_csv", separate_files=False)
                print("Data exported to CSV format in 'exported_data_csv' directory")
            elif format_type.lower() == "json":
                exporter = JSONExporter(self.data_repository)
                exporter.export_all_runs("exported_data_json", combined=True)
                print("Data exported to JSON format in 'exported_data_json' directory")
        except Exception as e:
            print(f"Error exporting data: {e}")

    def run(self):
        """Main simulation loop"""
        dt = 1.0 / FPS

        # Start initial data collection
        if self.data_collection_enabled:
            self.start_data_collection()

        while self.running:
            self.handle_events()

            if not self.paused:
                # Plan action for ego vehicle
                if self.current_env.ego_vehicle:
                    action = self.planner.plan_action(self.current_env.ego_vehicle)
                    self.current_env.ego_vehicle.set_action(action)

                    # Collect action data
                    if self.data_collection_enabled:
                        current_time = pygame.time.get_ticks() / 1000.0 - self.simulation_start_time
                        reason = "autonomous_decision"
                        self.data_manager.collect_action_data(
                            self.current_env.ego_vehicle,
                            action.name,
                            reason,
                            current_time
                        )

                # Update environment
                self.current_env.step(dt)

                # Collect simulation data
                if self.data_collection_enabled:
                    self.collect_simulation_data()

                # Simple AI for other vehicles
                for vehicle in self.current_env.vehicles:
                    if not vehicle.is_ego:
                        # Basic speed maintenance
                        speed = math.sqrt(vehicle.state.vx**2 + vehicle.state.vy**2)
                        if speed < DEFAULT_SPEEDS[0] - 5:
                            vehicle.state.acceleration = 1
                        elif speed > DEFAULT_SPEEDS[1]:
                            vehicle.state.acceleration = -1
                        else:
                            vehicle.state.acceleration = 0

            # Draw everything
            self.current_env.draw(self.screen)
            self.draw_hud()
            
            pygame.display.flip()
            self.clock.tick(FPS)

        # Clean up data collection
        if self.data_collection_enabled and self.current_run_id:
            self.data_repository.end_current_run()
            print("Final simulation run saved")

        pygame.quit()
        sys.exit()

def main():
    """Main entry point for the simulation"""
    print("Starting Autonomous Vehicle Simulation")
    print("Based on: 'Safety and Risk Analysis of Autonomous Vehicles'")
    print("=" * 50)

    simulation = Simulation()
    simulation.run()

if __name__ == "__main__":
    main()

"""
Data collectors for capturing simulation data in real-time.
"""

import time
import math
from typing import Dict, List, Optional, Any
from .repository import (
    DataRepository, VehicleSnapshot, EnvironmentSnapshot,
    CollisionEvent, ActionEvent
)


class BaseDataCollector:
    """Base class for data collectors"""

    def __init__(self, repository: DataRepository, collection_interval: float = 0.1):
        self.repository = repository
        self.collection_interval = collection_interval
        self.last_collection_time = 0.0
        self.enabled = True

    def should_collect(self, current_time: float) -> bool:
        """Check if it's time to collect data"""
        return (self.enabled and
                current_time - self.last_collection_time >= self.collection_interval)

    def update_collection_time(self, current_time: float):
        """Update the last collection time"""
        self.last_collection_time = current_time


class VehicleDataCollector(BaseDataCollector):
    """Collects vehicle state data"""

    def __init__(self, repository: DataRepository, collection_interval: float = 0.1):
        super().__init__(repository, collection_interval)
        self.vehicle_ids: Dict[Any, str] = {}  # Map vehicle objects to IDs

    def collect_vehicle_data(self, vehicles: List[Any], current_time: float):
        """Collect data from all vehicles"""
        if not self.should_collect(current_time):
            return

        for vehicle in vehicles:
            vehicle_id = self._get_vehicle_id(vehicle)

            # Calculate speed
            speed = math.sqrt(vehicle.state.vx**2 + vehicle.state.vy**2)

            snapshot = VehicleSnapshot(
                vehicle_id=vehicle_id,
                timestamp=current_time,
                position_x=vehicle.state.x,
                position_y=vehicle.state.y,
                velocity_x=vehicle.state.vx,
                velocity_y=vehicle.state.vy,
                acceleration=vehicle.state.acceleration,
                steering_angle=vehicle.state.steering_angle,
                heading=vehicle.state.heading,
                speed=speed,
                is_ego=vehicle.is_ego,
                lane_id=self._detect_lane(vehicle)
            )

            self.repository.add_vehicle_snapshot(snapshot)

        self.update_collection_time(current_time)

    def _get_vehicle_id(self, vehicle: Any) -> str:
        """Get or create a unique ID for a vehicle"""
        if vehicle not in self.vehicle_ids:
            # Create ID based on vehicle properties
            vehicle_type = "ego" if vehicle.is_ego else "other"
            vehicle_num = len([vid for vid in self.vehicle_ids.values()
                             if vid.startswith(vehicle_type)])
            self.vehicle_ids[vehicle] = f"{vehicle_type}_{vehicle_num}"

        return self.vehicle_ids[vehicle]

    def _detect_lane(self, vehicle: Any) -> Optional[int]:
        """Detect which lane the vehicle is in (simplified)"""
        # This is a simplified lane detection based on y-position
        # In a real system, this would be more sophisticated
        lane_width = 4.0  # meters

        # Assume lanes are centered around certain y-coordinates
        # This would need to be adapted based on the specific environment
        y_pos = vehicle.state.y

        # For highway environment (4 lanes)
        base_y = 200  # Approximate center of road in screen coordinates / PIXELS_PER_METER

        lane_centers = [base_y + (i - 1.5) * lane_width for i in range(4)]

        # Find closest lane center
        closest_lane = 0
        min_distance = float('inf')

        for i, center in enumerate(lane_centers):
            distance = abs(y_pos - center)
            if distance < min_distance:
                min_distance = distance
                closest_lane = i

        # Only return lane if vehicle is reasonably close to a lane center
        if min_distance < lane_width / 2:
            return closest_lane

        return None


class EnvironmentDataCollector(BaseDataCollector):
    """Collects environment state data"""

    def collect_environment_data(self, environment: Any, current_time: float):
        """Collect environment state data"""
        if not self.should_collect(current_time):
            return

        # Get ego vehicle ID if available
        ego_vehicle_id = None
        if environment.ego_vehicle:
            # Create a temporary vehicle collector to get the ID
            temp_collector = VehicleDataCollector(self.repository)
            ego_vehicle_id = temp_collector._get_vehicle_id(environment.ego_vehicle)

        snapshot = EnvironmentSnapshot(
            timestamp=current_time,
            environment_type=self._get_environment_type(environment),
            active_vehicles=len(environment.vehicles),
            collision_detected=environment.collision_detected,
            ego_vehicle_id=ego_vehicle_id
        )

        self.repository.add_environment_snapshot(snapshot)
        self.update_collection_time(current_time)

    def _get_environment_type(self, environment: Any) -> str:
        """Get the environment type name"""
        class_name = environment.__class__.__name__
        if "Highway" in class_name:
            return "highway"
        elif "Merging" in class_name:
            return "merging"
        elif "Roundabout" in class_name:
            return "roundabout"
        else:
            return "unknown"


class CollisionDataCollector(BaseDataCollector):
    """Collects collision event data"""

    def __init__(self, repository: DataRepository):
        super().__init__(repository, collection_interval=0.0)  # Immediate collection
        self.last_collision_state = False
        self.vehicle_collector = VehicleDataCollector(repository)

    def check_and_collect_collisions(self, environment: Any, current_time: float):
        """Check for new collisions and collect data"""
        current_collision_state = environment.collision_detected

        # Only collect if collision state changed from False to True
        if current_collision_state and not self.last_collision_state:
            self._collect_collision_event(environment, current_time)

        self.last_collision_state = current_collision_state

    def _collect_collision_event(self, environment: Any, current_time: float):
        """Collect collision event data"""
        if not environment.ego_vehicle:
            return

        # Find the colliding vehicles
        ego_vehicle = environment.ego_vehicle
        ego_rect = ego_vehicle.get_rect()

        colliding_vehicle = None
        for vehicle in environment.vehicles:
            if vehicle != ego_vehicle:
                if ego_rect.colliderect(vehicle.get_rect()):
                    colliding_vehicle = vehicle
                    break

        if colliding_vehicle:
            # Calculate collision point (midpoint between vehicle centers)
            collision_x = (ego_vehicle.state.x + colliding_vehicle.state.x) / 2
            collision_y = (ego_vehicle.state.y + colliding_vehicle.state.y) / 2

            # Calculate relative speed
            ego_speed = math.sqrt(ego_vehicle.state.vx**2 + ego_vehicle.state.vy**2)
            other_speed = math.sqrt(colliding_vehicle.state.vx**2 + colliding_vehicle.state.vy**2)
            relative_speed = abs(ego_speed - other_speed)

            # Get vehicle IDs
            ego_id = self.vehicle_collector._get_vehicle_id(ego_vehicle)
            other_id = self.vehicle_collector._get_vehicle_id(colliding_vehicle)

            collision_event = CollisionEvent(
                timestamp=current_time,
                vehicle1_id=ego_id,
                vehicle2_id=other_id,
                collision_point_x=collision_x,
                collision_point_y=collision_y,
                relative_speed=relative_speed,
                environment_type=self._get_environment_type(environment)
            )

            self.repository.add_collision_event(collision_event)

    def _get_environment_type(self, environment: Any) -> str:
        """Get the environment type name"""
        class_name = environment.__class__.__name__
        if "Highway" in class_name:
            return "highway"
        elif "Merging" in class_name:
            return "merging"
        elif "Roundabout" in class_name:
            return "roundabout"
        else:
            return "unknown"


class ActionDataCollector(BaseDataCollector):
    """Collects vehicle action/decision data"""

    def __init__(self, repository: DataRepository):
        super().__init__(repository, collection_interval=0.0)  # Immediate collection
        self.vehicle_actions: Dict[str, str] = {}  # Track last action per vehicle
        self.vehicle_collector = VehicleDataCollector(repository)

    def collect_action(self, vehicle: Any, action: str, reason: str, current_time: float):
        """Collect vehicle action data"""
        vehicle_id = self.vehicle_collector._get_vehicle_id(vehicle)
        previous_action = self.vehicle_actions.get(vehicle_id, "IDLE")

        # Only collect if action changed
        if action != previous_action:
            action_event = ActionEvent(
                timestamp=current_time,
                vehicle_id=vehicle_id,
                action=action,
                previous_action=previous_action,
                reason=reason
            )

            self.repository.add_action_event(action_event)
            self.vehicle_actions[vehicle_id] = action


class DataCollectionManager:
    """Manages all data collectors"""

    def __init__(self, repository: DataRepository):
        self.repository = repository
        self.vehicle_collector = VehicleDataCollector(repository)
        self.environment_collector = EnvironmentDataCollector(repository)
        self.collision_collector = CollisionDataCollector(repository)
        self.action_collector = ActionDataCollector(repository)

    def collect_all_data(self, environment: Any, current_time: float):
        """Collect all types of data"""
        # Collect vehicle data
        self.vehicle_collector.collect_vehicle_data(environment.vehicles, current_time)

        # Collect environment data
        self.environment_collector.collect_environment_data(environment, current_time)

        # Check for collisions
        self.collision_collector.check_and_collect_collisions(environment, current_time)

    def collect_action_data(self, vehicle: Any, action: str, reason: str, current_time: float):
        """Collect action data"""
        self.action_collector.collect_action(vehicle, action, reason, current_time)

    def set_collection_intervals(self, vehicle_interval: float = 0.1,
                                environment_interval: float = 0.1):
        """Set collection intervals for different data types"""
        self.vehicle_collector.collection_interval = vehicle_interval
        self.environment_collector.collection_interval = environment_interval

    def enable_collection(self, enabled: bool = True):
        """Enable or disable all data collection"""
        self.vehicle_collector.enabled = enabled
        self.environment_collector.enabled = enabled
        self.collision_collector.enabled = enabled
        self.action_collector.enabled = enabled
"""
Data repository for storing and managing simulation data.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import threading
import json
import os


@dataclass
class VehicleSnapshot:
    """Single timestep data for a vehicle"""
    vehicle_id: str
    timestamp: float
    position_x: float
    position_y: float
    velocity_x: float
    velocity_y: float
    acceleration: float
    steering_angle: float
    heading: float
    speed: float
    is_ego: bool
    lane_id: Optional[int] = None


@dataclass
class EnvironmentSnapshot:
    """Single timestep data for the environment"""
    timestamp: float
    environment_type: str
    active_vehicles: int
    collision_detected: bool
    ego_vehicle_id: Optional[str] = None


@dataclass
class CollisionEvent:
    """Data for collision events"""
    timestamp: float
    vehicle1_id: str
    vehicle2_id: str
    collision_point_x: float
    collision_point_y: float
    relative_speed: float
    environment_type: str


@dataclass
class ActionEvent:
    """Data for vehicle action events"""
    timestamp: float
    vehicle_id: str
    action: str
    previous_action: str
    reason: str


@dataclass
class SimulationRun:
    """Complete data for a single simulation run"""
    run_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    environment_type: str = ""
    total_duration: float = 0.0
    vehicle_snapshots: List[VehicleSnapshot] = field(default_factory=list)
    environment_snapshots: List[EnvironmentSnapshot] = field(default_factory=list)
    collision_events: List[CollisionEvent] = field(default_factory=list)
    action_events: List[ActionEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataRepository:
    """Central repository for simulation data storage and retrieval"""

    def __init__(self, storage_path: str = "simulation_data"):
        self.storage_path = storage_path
        self.current_run: Optional[SimulationRun] = None
        self.runs: Dict[str, SimulationRun] = {}
        self._lock = threading.Lock()

        # Create storage directory
        os.makedirs(storage_path, exist_ok=True)

        # Load existing runs
        self._load_existing_runs()

    def start_new_run(self, environment_type: str, metadata: Dict[str, Any] = None) -> str:
        """Start a new simulation run"""
        with self._lock:
            run_id = str(uuid.uuid4())
            self.current_run = SimulationRun(
                run_id=run_id,
                start_time=datetime.now(),
                environment_type=environment_type,
                metadata=metadata or {}
            )
            self.runs[run_id] = self.current_run
            return run_id

    def end_current_run(self):
        """End the current simulation run"""
        with self._lock:
            if self.current_run:
                self.current_run.end_time = datetime.now()
                if self.current_run.start_time and self.current_run.end_time:
                    self.current_run.total_duration = (
                        self.current_run.end_time - self.current_run.start_time
                    ).total_seconds()
                self._save_run(self.current_run)
                self.current_run = None

    def add_vehicle_snapshot(self, snapshot: VehicleSnapshot):
        """Add vehicle data snapshot"""
        with self._lock:
            if self.current_run:
                self.current_run.vehicle_snapshots.append(snapshot)

    def add_environment_snapshot(self, snapshot: EnvironmentSnapshot):
        """Add environment data snapshot"""
        with self._lock:
            if self.current_run:
                self.current_run.environment_snapshots.append(snapshot)

    def add_collision_event(self, event: CollisionEvent):
        """Add collision event"""
        with self._lock:
            if self.current_run:
                self.current_run.collision_events.append(event)

    def add_action_event(self, event: ActionEvent):
        """Add action event"""
        with self._lock:
            if self.current_run:
                self.current_run.action_events.append(event)

    def get_run(self, run_id: str) -> Optional[SimulationRun]:
        """Get simulation run by ID"""
        return self.runs.get(run_id)

    def get_all_runs(self) -> List[SimulationRun]:
        """Get all simulation runs"""
        return list(self.runs.values())

    def get_runs_by_environment(self, environment_type: str) -> List[SimulationRun]:
        """Get runs filtered by environment type"""
        return [run for run in self.runs.values()
                if run.environment_type == environment_type]

    def get_collision_statistics(self) -> Dict[str, Any]:
        """Get collision statistics across all runs"""
        total_runs = len(self.runs)
        runs_with_collisions = len([run for run in self.runs.values()
                                   if run.collision_events])
        total_collisions = sum(len(run.collision_events) for run in self.runs.values())

        return {
            "total_runs": total_runs,
            "runs_with_collisions": runs_with_collisions,
            "collision_rate": runs_with_collisions / total_runs if total_runs > 0 else 0,
            "total_collisions": total_collisions,
            "average_collisions_per_run": total_collisions / total_runs if total_runs > 0 else 0
        }

    def _save_run(self, run: SimulationRun):
        """Save simulation run to disk"""
        filename = os.path.join(self.storage_path, f"run_{run.run_id}.json")

        # Convert dataclass to dict for JSON serialization
        run_data = {
            "run_id": run.run_id,
            "start_time": run.start_time.isoformat() if run.start_time else None,
            "end_time": run.end_time.isoformat() if run.end_time else None,
            "environment_type": run.environment_type,
            "total_duration": run.total_duration,
            "metadata": run.metadata,
            "vehicle_snapshots": [
                {
                    "vehicle_id": vs.vehicle_id,
                    "timestamp": vs.timestamp,
                    "position_x": vs.position_x,
                    "position_y": vs.position_y,
                    "velocity_x": vs.velocity_x,
                    "velocity_y": vs.velocity_y,
                    "acceleration": vs.acceleration,
                    "steering_angle": vs.steering_angle,
                    "heading": vs.heading,
                    "speed": vs.speed,
                    "is_ego": vs.is_ego,
                    "lane_id": vs.lane_id
                }
                for vs in run.vehicle_snapshots
            ],
            "environment_snapshots": [
                {
                    "timestamp": es.timestamp,
                    "environment_type": es.environment_type,
                    "active_vehicles": es.active_vehicles,
                    "collision_detected": es.collision_detected,
                    "ego_vehicle_id": es.ego_vehicle_id
                }
                for es in run.environment_snapshots
            ],
            "collision_events": [
                {
                    "timestamp": ce.timestamp,
                    "vehicle1_id": ce.vehicle1_id,
                    "vehicle2_id": ce.vehicle2_id,
                    "collision_point_x": ce.collision_point_x,
                    "collision_point_y": ce.collision_point_y,
                    "relative_speed": ce.relative_speed,
                    "environment_type": ce.environment_type
                }
                for ce in run.collision_events
            ],
            "action_events": [
                {
                    "timestamp": ae.timestamp,
                    "vehicle_id": ae.vehicle_id,
                    "action": ae.action,
                    "previous_action": ae.previous_action,
                    "reason": ae.reason
                }
                for ae in run.action_events
            ]
        }

        with open(filename, 'w') as f:
            json.dump(run_data, f, indent=2)

    def _load_existing_runs(self):
        """Load existing simulation runs from disk"""
        if not os.path.exists(self.storage_path):
            return

        for filename in os.listdir(self.storage_path):
            if filename.startswith("run_") and filename.endswith(".json"):
                filepath = os.path.join(self.storage_path, filename)
                try:
                    with open(filepath, 'r') as f:
                        run_data = json.load(f)

                    # Convert back to dataclass
                    run = SimulationRun(
                        run_id=run_data["run_id"],
                        start_time=datetime.fromisoformat(run_data["start_time"]) if run_data["start_time"] else None,
                        end_time=datetime.fromisoformat(run_data["end_time"]) if run_data["end_time"] else None,
                        environment_type=run_data["environment_type"],
                        total_duration=run_data["total_duration"],
                        metadata=run_data["metadata"],
                        vehicle_snapshots=[
                            VehicleSnapshot(**vs) for vs in run_data["vehicle_snapshots"]
                        ],
                        environment_snapshots=[
                            EnvironmentSnapshot(**es) for es in run_data["environment_snapshots"]
                        ],
                        collision_events=[
                            CollisionEvent(**ce) for ce in run_data["collision_events"]
                        ],
                        action_events=[
                            ActionEvent(**ae) for ae in run_data["action_events"]
                        ]
                    )

                    self.runs[run.run_id] = run

                except Exception as e:
                    print(f"Error loading run from {filename}: {e}")


class SimulationData:
    """Convenience class for accessing simulation data"""

    def __init__(self, repository: DataRepository):
        self.repository = repository

    def get_vehicle_trajectory(self, run_id: str, vehicle_id: str) -> List[VehicleSnapshot]:
        """Get complete trajectory for a specific vehicle"""
        run = self.repository.get_run(run_id)
        if not run:
            return []

        return [vs for vs in run.vehicle_snapshots if vs.vehicle_id == vehicle_id]

    def get_ego_vehicle_data(self, run_id: str) -> List[VehicleSnapshot]:
        """Get ego vehicle data for a run"""
        run = self.repository.get_run(run_id)
        if not run:
            return []

        return [vs for vs in run.vehicle_snapshots if vs.is_ego]

    def get_speed_profile(self, run_id: str, vehicle_id: str) -> List[tuple]:
        """Get speed profile as (timestamp, speed) pairs"""
        trajectory = self.get_vehicle_trajectory(run_id, vehicle_id)
        return [(vs.timestamp, vs.speed) for vs in trajectory]

    def get_collision_analysis(self, run_id: str) -> Dict[str, Any]:
        """Get detailed collision analysis for a run"""
        run = self.repository.get_run(run_id)
        if not run:
            return {}

        analysis = {
            "total_collisions": len(run.collision_events),
            "collision_times": [ce.timestamp for ce in run.collision_events],
            "collision_locations": [(ce.collision_point_x, ce.collision_point_y)
                                   for ce in run.collision_events],
            "collision_speeds": [ce.relative_speed for ce in run.collision_events]
        }

        if run.collision_events:
            analysis["first_collision_time"] = min(ce.timestamp for ce in run.collision_events)
            analysis["average_collision_speed"] = sum(ce.relative_speed for ce in run.collision_events) / len(run.collision_events)

        return analysis
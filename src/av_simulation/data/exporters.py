"""
Data exporters for converting simulation data to various formats.
"""

import csv
import json
import os
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
from .repository import DataRepository, SimulationRun


class BaseExporter:
    """Base class for data exporters"""

    def __init__(self, repository: DataRepository):
        self.repository = repository

    def export_run(self, run_id: str, output_path: str, **kwargs):
        """Export a single simulation run"""
        raise NotImplementedError

    def export_all_runs(self, output_path: str, **kwargs):
        """Export all simulation runs"""
        raise NotImplementedError


class CSVExporter(BaseExporter):
    """Export simulation data to CSV format"""

    def export_run(self, run_id: str, output_path: str, include_metadata: bool = True):
        """Export a single run to CSV files"""
        run = self.repository.get_run(run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")

        # Create output directory
        run_dir = os.path.join(output_path, f"run_{run_id}")
        os.makedirs(run_dir, exist_ok=True)

        # Export vehicle data
        self._export_vehicle_data_csv(run, run_dir)

        # Export environment data
        self._export_environment_data_csv(run, run_dir)

        # Export collision events
        self._export_collision_events_csv(run, run_dir)

        # Export action events
        self._export_action_events_csv(run, run_dir)

        # Export metadata
        if include_metadata:
            self._export_metadata_csv(run, run_dir)

    def export_all_runs(self, output_path: str, separate_files: bool = True):
        """Export all runs to CSV"""
        if separate_files:
            for run_id in self.repository.runs.keys():
                self.export_run(run_id, output_path)
        else:
            # Combined export
            self._export_combined_csv(output_path)

    def _export_vehicle_data_csv(self, run: SimulationRun, output_dir: str):
        """Export vehicle snapshots to CSV"""
        if not run.vehicle_snapshots:
            return

        filename = os.path.join(output_dir, "vehicle_data.csv")
        fieldnames = [
            'vehicle_id', 'timestamp', 'position_x', 'position_y',
            'velocity_x', 'velocity_y', 'acceleration', 'steering_angle',
            'heading', 'speed', 'is_ego', 'lane_id'
        ]

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for snapshot in run.vehicle_snapshots:
                writer.writerow({
                    'vehicle_id': snapshot.vehicle_id,
                    'timestamp': snapshot.timestamp,
                    'position_x': snapshot.position_x,
                    'position_y': snapshot.position_y,
                    'velocity_x': snapshot.velocity_x,
                    'velocity_y': snapshot.velocity_y,
                    'acceleration': snapshot.acceleration,
                    'steering_angle': snapshot.steering_angle,
                    'heading': snapshot.heading,
                    'speed': snapshot.speed,
                    'is_ego': snapshot.is_ego,
                    'lane_id': snapshot.lane_id
                })

    def _export_environment_data_csv(self, run: SimulationRun, output_dir: str):
        """Export environment snapshots to CSV"""
        if not run.environment_snapshots:
            return

        filename = os.path.join(output_dir, "environment_data.csv")
        fieldnames = [
            'timestamp', 'environment_type', 'active_vehicles',
            'collision_detected', 'ego_vehicle_id'
        ]

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for snapshot in run.environment_snapshots:
                writer.writerow({
                    'timestamp': snapshot.timestamp,
                    'environment_type': snapshot.environment_type,
                    'active_vehicles': snapshot.active_vehicles,
                    'collision_detected': snapshot.collision_detected,
                    'ego_vehicle_id': snapshot.ego_vehicle_id
                })

    def _export_collision_events_csv(self, run: SimulationRun, output_dir: str):
        """Export collision events to CSV"""
        if not run.collision_events:
            return

        filename = os.path.join(output_dir, "collision_events.csv")
        fieldnames = [
            'timestamp', 'vehicle1_id', 'vehicle2_id',
            'collision_point_x', 'collision_point_y',
            'relative_speed', 'environment_type'
        ]

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for event in run.collision_events:
                writer.writerow({
                    'timestamp': event.timestamp,
                    'vehicle1_id': event.vehicle1_id,
                    'vehicle2_id': event.vehicle2_id,
                    'collision_point_x': event.collision_point_x,
                    'collision_point_y': event.collision_point_y,
                    'relative_speed': event.relative_speed,
                    'environment_type': event.environment_type
                })

    def _export_action_events_csv(self, run: SimulationRun, output_dir: str):
        """Export action events to CSV"""
        if not run.action_events:
            return

        filename = os.path.join(output_dir, "action_events.csv")
        fieldnames = [
            'timestamp', 'vehicle_id', 'action',
            'previous_action', 'reason'
        ]

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for event in run.action_events:
                writer.writerow({
                    'timestamp': event.timestamp,
                    'vehicle_id': event.vehicle_id,
                    'action': event.action,
                    'previous_action': event.previous_action,
                    'reason': event.reason
                })

    def _export_metadata_csv(self, run: SimulationRun, output_dir: str):
        """Export run metadata to CSV"""
        filename = os.path.join(output_dir, "metadata.csv")

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['property', 'value'])
            writer.writerow(['run_id', run.run_id])
            writer.writerow(['start_time', run.start_time.isoformat() if run.start_time else None])
            writer.writerow(['end_time', run.end_time.isoformat() if run.end_time else None])
            writer.writerow(['environment_type', run.environment_type])
            writer.writerow(['total_duration', run.total_duration])

            for key, value in run.metadata.items():
                writer.writerow([f'metadata_{key}', value])

    def _export_combined_csv(self, output_path: str):
        """Export all runs to combined CSV files"""
        os.makedirs(output_path, exist_ok=True)

        # Combine all vehicle data
        all_vehicle_data = []
        all_environment_data = []
        all_collision_events = []
        all_action_events = []

        for run in self.repository.get_all_runs():
            for snapshot in run.vehicle_snapshots:
                row = {
                    'run_id': run.run_id,
                    'vehicle_id': snapshot.vehicle_id,
                    'timestamp': snapshot.timestamp,
                    'position_x': snapshot.position_x,
                    'position_y': snapshot.position_y,
                    'velocity_x': snapshot.velocity_x,
                    'velocity_y': snapshot.velocity_y,
                    'acceleration': snapshot.acceleration,
                    'steering_angle': snapshot.steering_angle,
                    'heading': snapshot.heading,
                    'speed': snapshot.speed,
                    'is_ego': snapshot.is_ego,
                    'lane_id': snapshot.lane_id
                }
                all_vehicle_data.append(row)

            for snapshot in run.environment_snapshots:
                row = {
                    'run_id': run.run_id,
                    'timestamp': snapshot.timestamp,
                    'environment_type': snapshot.environment_type,
                    'active_vehicles': snapshot.active_vehicles,
                    'collision_detected': snapshot.collision_detected,
                    'ego_vehicle_id': snapshot.ego_vehicle_id
                }
                all_environment_data.append(row)

            for event in run.collision_events:
                row = {
                    'run_id': run.run_id,
                    'timestamp': event.timestamp,
                    'vehicle1_id': event.vehicle1_id,
                    'vehicle2_id': event.vehicle2_id,
                    'collision_point_x': event.collision_point_x,
                    'collision_point_y': event.collision_point_y,
                    'relative_speed': event.relative_speed,
                    'environment_type': event.environment_type
                }
                all_collision_events.append(row)

            for event in run.action_events:
                row = {
                    'run_id': run.run_id,
                    'timestamp': event.timestamp,
                    'vehicle_id': event.vehicle_id,
                    'action': event.action,
                    'previous_action': event.previous_action,
                    'reason': event.reason
                }
                all_action_events.append(row)

        # Write combined files
        if all_vehicle_data:
            df = pd.DataFrame(all_vehicle_data)
            df.to_csv(os.path.join(output_path, "all_vehicle_data.csv"), index=False)

        if all_environment_data:
            df = pd.DataFrame(all_environment_data)
            df.to_csv(os.path.join(output_path, "all_environment_data.csv"), index=False)

        if all_collision_events:
            df = pd.DataFrame(all_collision_events)
            df.to_csv(os.path.join(output_path, "all_collision_events.csv"), index=False)

        if all_action_events:
            df = pd.DataFrame(all_action_events)
            df.to_csv(os.path.join(output_path, "all_action_events.csv"), index=False)


class JSONExporter(BaseExporter):
    """Export simulation data to JSON format"""

    def export_run(self, run_id: str, output_path: str):
        """Export a single run to JSON"""
        run = self.repository.get_run(run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")

        os.makedirs(output_path, exist_ok=True)
        filename = os.path.join(output_path, f"run_{run_id}.json")

        # Convert run to dictionary
        run_data = self._run_to_dict(run)

        with open(filename, 'w') as f:
            json.dump(run_data, f, indent=2, default=str)

    def export_all_runs(self, output_path: str, combined: bool = False):
        """Export all runs to JSON"""
        os.makedirs(output_path, exist_ok=True)

        if combined:
            # Export all runs in a single file
            all_runs = {}
            for run_id, run in self.repository.runs.items():
                all_runs[run_id] = self._run_to_dict(run)

            filename = os.path.join(output_path, "all_runs.json")
            with open(filename, 'w') as f:
                json.dump(all_runs, f, indent=2, default=str)
        else:
            # Export each run separately
            for run_id in self.repository.runs.keys():
                self.export_run(run_id, output_path)

    def _run_to_dict(self, run: SimulationRun) -> Dict[str, Any]:
        """Convert SimulationRun to dictionary"""
        return {
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


class HDF5Exporter(BaseExporter):
    """Export simulation data to HDF5 format for large datasets"""

    def __init__(self, repository: DataRepository):
        super().__init__(repository)
        try:
            import h5py
            self.h5py = h5py
        except ImportError:
            raise ImportError("h5py is required for HDF5 export. Install with: pip install h5py")

    def export_run(self, run_id: str, output_path: str):
        """Export a single run to HDF5"""
        run = self.repository.get_run(run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")

        os.makedirs(output_path, exist_ok=True)
        filename = os.path.join(output_path, f"run_{run_id}.h5")

        with self.h5py.File(filename, 'w') as f:
            # Metadata
            meta_group = f.create_group('metadata')
            meta_group.attrs['run_id'] = run.run_id
            meta_group.attrs['start_time'] = str(run.start_time) if run.start_time else ""
            meta_group.attrs['end_time'] = str(run.end_time) if run.end_time else ""
            meta_group.attrs['environment_type'] = run.environment_type
            meta_group.attrs['total_duration'] = run.total_duration

            # Vehicle data
            if run.vehicle_snapshots:
                vehicle_group = f.create_group('vehicle_data')
                self._export_vehicle_data_h5(run.vehicle_snapshots, vehicle_group)

            # Environment data
            if run.environment_snapshots:
                env_group = f.create_group('environment_data')
                self._export_environment_data_h5(run.environment_snapshots, env_group)

            # Collision events
            if run.collision_events:
                collision_group = f.create_group('collision_events')
                self._export_collision_events_h5(run.collision_events, collision_group)

    def export_all_runs(self, output_path: str):
        """Export all runs to HDF5"""
        for run_id in self.repository.runs.keys():
            self.export_run(run_id, output_path)

    def _export_vehicle_data_h5(self, snapshots: List, group):
        """Export vehicle snapshots to HDF5 group"""
        if not snapshots:
            return

        # Create datasets for each field
        n_samples = len(snapshots)

        # String data
        vehicle_ids = [s.vehicle_id.encode('utf-8') for s in snapshots]
        group.create_dataset('vehicle_id', data=vehicle_ids)

        # Numeric data
        timestamps = [s.timestamp for s in snapshots]
        group.create_dataset('timestamp', data=timestamps)

        positions_x = [s.position_x for s in snapshots]
        group.create_dataset('position_x', data=positions_x)

        positions_y = [s.position_y for s in snapshots]
        group.create_dataset('position_y', data=positions_y)

        velocities_x = [s.velocity_x for s in snapshots]
        group.create_dataset('velocity_x', data=velocities_x)

        velocities_y = [s.velocity_y for s in snapshots]
        group.create_dataset('velocity_y', data=velocities_y)

        accelerations = [s.acceleration for s in snapshots]
        group.create_dataset('acceleration', data=accelerations)

        steering_angles = [s.steering_angle for s in snapshots]
        group.create_dataset('steering_angle', data=steering_angles)

        headings = [s.heading for s in snapshots]
        group.create_dataset('heading', data=headings)

        speeds = [s.speed for s in snapshots]
        group.create_dataset('speed', data=speeds)

        is_ego = [s.is_ego for s in snapshots]
        group.create_dataset('is_ego', data=is_ego)

        lane_ids = [s.lane_id if s.lane_id is not None else -1 for s in snapshots]
        group.create_dataset('lane_id', data=lane_ids)

    def _export_environment_data_h5(self, snapshots: List, group):
        """Export environment snapshots to HDF5 group"""
        if not snapshots:
            return

        timestamps = [s.timestamp for s in snapshots]
        group.create_dataset('timestamp', data=timestamps)

        env_types = [s.environment_type.encode('utf-8') for s in snapshots]
        group.create_dataset('environment_type', data=env_types)

        active_vehicles = [s.active_vehicles for s in snapshots]
        group.create_dataset('active_vehicles', data=active_vehicles)

        collision_detected = [s.collision_detected for s in snapshots]
        group.create_dataset('collision_detected', data=collision_detected)

    def _export_collision_events_h5(self, events: List, group):
        """Export collision events to HDF5 group"""
        if not events:
            return

        timestamps = [e.timestamp for e in events]
        group.create_dataset('timestamp', data=timestamps)

        vehicle1_ids = [e.vehicle1_id.encode('utf-8') for e in events]
        group.create_dataset('vehicle1_id', data=vehicle1_ids)

        vehicle2_ids = [e.vehicle2_id.encode('utf-8') for e in events]
        group.create_dataset('vehicle2_id', data=vehicle2_ids)

        collision_points_x = [e.collision_point_x for e in events]
        group.create_dataset('collision_point_x', data=collision_points_x)

        collision_points_y = [e.collision_point_y for e in events]
        group.create_dataset('collision_point_y', data=collision_points_y)

        relative_speeds = [e.relative_speed for e in events]
        group.create_dataset('relative_speed', data=relative_speeds)
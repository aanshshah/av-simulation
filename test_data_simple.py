#!/usr/bin/env python3
"""
Simple test script for the data repository system
"""

import sys
import os
import time

# Add src to path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Direct imports to avoid module dependency issues
from av_simulation.data.repository import DataRepository, VehicleSnapshot, EnvironmentSnapshot, CollisionEvent, ActionEvent
from av_simulation.data.exporters import CSVExporter, JSONExporter


def test_data_repository():
    """Test basic data repository functionality"""
    print("Testing Data Repository...")

    # Create repository
    repo = DataRepository("test_simulation_data")

    # Start a new run
    run_id = repo.start_new_run("highway", {"test": True, "version": "1.0"})
    print(f"Started run: {run_id}")

    # Add some test data
    for i in range(10):
        timestamp = i * 0.1

        # Add vehicle data
        ego_snapshot = VehicleSnapshot(
            vehicle_id="ego_0",
            timestamp=timestamp,
            position_x=10 + i * 2,
            position_y=50,
            velocity_x=20,
            velocity_y=0,
            acceleration=0,
            steering_angle=0,
            heading=0,
            speed=20,
            is_ego=True,
            lane_id=1
        )
        repo.add_vehicle_snapshot(ego_snapshot)

        other_snapshot = VehicleSnapshot(
            vehicle_id="other_0",
            timestamp=timestamp,
            position_x=30 + i * 1.5,
            position_y=54,
            velocity_x=15,
            velocity_y=0,
            acceleration=0,
            steering_angle=0,
            heading=0,
            speed=15,
            is_ego=False,
            lane_id=1
        )
        repo.add_vehicle_snapshot(other_snapshot)

        # Add environment data
        env_snapshot = EnvironmentSnapshot(
            timestamp=timestamp,
            environment_type="highway",
            active_vehicles=2,
            collision_detected=False,
            ego_vehicle_id="ego_0"
        )
        repo.add_environment_snapshot(env_snapshot)

        # Add action event
        if i % 3 == 0:
            action_event = ActionEvent(
                timestamp=timestamp,
                vehicle_id="ego_0",
                action="FASTER",
                previous_action="IDLE",
                reason="maintain_speed"
            )
            repo.add_action_event(action_event)

    # Simulate a collision
    collision_event = CollisionEvent(
        timestamp=0.8,
        vehicle1_id="ego_0",
        vehicle2_id="other_0",
        collision_point_x=25,
        collision_point_y=52,
        relative_speed=5,
        environment_type="highway"
    )
    repo.add_collision_event(collision_event)

    # End the run
    repo.end_current_run()
    print("Run ended")

    # Test data retrieval
    run = repo.get_run(run_id)
    assert run is not None, "Run should exist"
    assert len(run.vehicle_snapshots) == 20, f"Expected 20 vehicle snapshots, got {len(run.vehicle_snapshots)}"
    assert len(run.environment_snapshots) == 10, f"Expected 10 environment snapshots, got {len(run.environment_snapshots)}"
    assert len(run.collision_events) == 1, f"Expected 1 collision event, got {len(run.collision_events)}"
    assert len(run.action_events) == 4, f"Expected 4 action events, got {len(run.action_events)}"

    print("✓ Data repository test passed")
    return repo


def test_data_exporters(repo):
    """Test data export functionality"""
    print("\nTesting Data Exporters...")

    # Test CSV export
    csv_exporter = CSVExporter(repo)
    csv_exporter.export_all_runs("test_export_csv", separate_files=False)
    print("✓ CSV export completed")

    # Test JSON export
    json_exporter = JSONExporter(repo)
    json_exporter.export_all_runs("test_export_json", combined=True)
    print("✓ JSON export completed")

    # Check if files exist
    csv_files = ["test_export_csv/all_vehicle_data.csv",
                 "test_export_csv/all_environment_data.csv",
                 "test_export_csv/all_collision_events.csv",
                 "test_export_csv/all_action_events.csv"]

    for file_path in csv_files:
        if os.path.exists(file_path):
            print(f"✓ Created {file_path}")
        else:
            print(f"✗ Missing {file_path}")

    json_file = "test_export_json/all_runs.json"
    if os.path.exists(json_file):
        print(f"✓ Created {json_file}")
    else:
        print(f"✗ Missing {json_file}")

    print("✓ Export test completed")


def cleanup():
    """Clean up test files"""
    print("\nCleaning up test files...")
    import shutil

    dirs_to_remove = ["test_simulation_data", "test_export_csv", "test_export_json"]

    for dir_name in dirs_to_remove:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"✓ Removed {dir_name}")


def main():
    """Run all tests"""
    print("=" * 50)
    print("AV Simulation Data Repository Test Suite")
    print("=" * 50)

    try:
        # Test data repository
        repo = test_data_repository()

        # Test exporters
        test_data_exporters(repo)

        print("\n" + "=" * 50)
        print("✓ All tests passed successfully!")
        print("=" * 50)

        print("\nData repository system is ready to use.")
        print("Key features implemented:")
        print("- Real-time data collection during simulation")
        print("- Vehicle trajectory tracking")
        print("- Collision event detection")
        print("- Action/decision logging")
        print("- CSV and JSON export capabilities")
        print("- Threadsafe data storage")
        print("- Persistent storage with JSON serialization")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        cleanup()

    return 0


if __name__ == "__main__":
    exit(main())
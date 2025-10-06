"""
Data collection and repository module for AV simulation.
"""

from .repository import DataRepository, SimulationData
from .collectors import VehicleDataCollector, EnvironmentDataCollector, CollisionDataCollector
from .exporters import CSVExporter, JSONExporter, HDF5Exporter
from .dataset_info import (
    DatasetMetadata,
    ModalityDescriptor,
    ScenarioDescriptor,
    AnnotationDescriptor,
    get_dataset_metadata,
    describe_dataset,
)

__all__ = [
    "DataRepository",
    "SimulationData",
    "VehicleDataCollector",
    "EnvironmentDataCollector",
    "CollisionDataCollector",
    "CSVExporter",
    "JSONExporter",
    "HDF5Exporter",
    "DatasetMetadata",
    "ModalityDescriptor",
    "ScenarioDescriptor",
    "AnnotationDescriptor",
    "get_dataset_metadata",
    "describe_dataset",
]

"""Dataset metadata definitions for the Dixit et al. (2021) AV safety dataset.

This module captures the key characteristics of the hybrid real/simulated dataset
referenced throughout the project so that notebooks and scripts can reason about
available modalities, annotation targets, and benchmark scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass(frozen=True)
class ModalityDescriptor:
    """Describe a single sensing modality that is available in the dataset."""

    name: str
    source: str
    format: str
    description: str


@dataclass(frozen=True)
class ScenarioDescriptor:
    """Describe a real or simulated driving scenario."""

    name: str
    environment: str
    highlights: List[str]


@dataclass(frozen=True)
class AnnotationDescriptor:
    """Describe the semantic labels packaged with the dataset."""

    label: str
    task: str
    description: str


@dataclass(frozen=True)
class DatasetMetadata:
    """Aggregate description of the Dixit et al. (2021) dataset."""

    citation: str
    dataset_url: str
    modalities: List[ModalityDescriptor]
    annotations: List[AnnotationDescriptor]
    scenarios: List[ScenarioDescriptor]
    notes: List[str]

    def to_dict(self) -> Dict[str, object]:
        """Convert the metadata object into a plain dictionary."""

        return asdict(self)


def get_dataset_metadata() -> DatasetMetadata:
    """Return a structured summary of the dataset referenced in the paper."""

    modalities = [
        ModalityDescriptor(
            name="Camera RGB",
            source="On-vehicle stereo rig",
            format="1080p MP4 sequences (30 FPS)",
            description=(
                "Forward-facing roadway footage spanning daylight, dusk, and rainfall "
                "conditions. Used for supervised lane detection, obstacle identification, "
                "and traffic participant classification tasks."
            ),
        ),
        ModalityDescriptor(
            name="LiDAR point clouds",
            source="Velodyne HDL-32E",
            format=".pcd sequences synchronised with video",
            description=(
                "3D range scans aligned to camera frames for obstacle localisation and "
                "cross-modal sensor fusion exercises."
            ),
        ),
        ModalityDescriptor(
            name="Simulated RGB",
            source="OpenCV + Pygame renderers",
            format="PNG frame dumps per scenario",
            description=(
                "Domain-randomised imagery generated for controlled evaluation of highway, "
                "lane-merge, and roundabout encounters as described by Dixit et al."
            ),
        ),
    ]

    annotations = [
        AnnotationDescriptor(
            label="lane_marking",
            task="Semantic segmentation",
            description="Pixel-level masks for centre, left, and right lane boundaries.",
        ),
        AnnotationDescriptor(
            label="vehicle",
            task="Bounding box / instance masks",
            description="Annotations for cars, trucks, buses, and motorcycles.",
        ),
        AnnotationDescriptor(
            label="pedestrian",
            task="Bounding box",
            description="Human participants crossing or walking alongside the roadway.",
        ),
        AnnotationDescriptor(
            label="curved_road",
            task="Scene classification",
            description="Frame-level indicator for curved geometry useful for trajectory planning.",
        ),
        AnnotationDescriptor(
            label="ambiguous_weather",
            task="Domain tag",
            description="Metadata flag capturing low-visibility conditions (fog, heavy rain, dusk).",
        ),
    ]

    scenarios = [
        ScenarioDescriptor(
            name="Urban arterial daytime",
            environment="Real-world camera + LiDAR",
            highlights=[
                "Dense traffic with frequent pedestrian crossings",
                "Lane tracking under strong shadows",
                "Mixed vehicle classes"
            ],
        ),
        ScenarioDescriptor(
            name="Adverse weather nighttime",
            environment="Real-world camera + LiDAR",
            highlights=[
                "Heavy rain and specular reflections",
                "Reduced signal-to-noise ratio for lane markings",
                "High-beam glare management"
            ],
        ),
        ScenarioDescriptor(
            name="Highway (simulated)",
            environment="OpenCV + Pygame",
            highlights=[
                "Four-lane highway with controlled traffic flow",
                "Safe following distance experiments",
                "Hazard injection via sudden braking"
            ],
        ),
        ScenarioDescriptor(
            name="Lane merge (simulated)",
            environment="OpenCV + Pygame",
            highlights=[
                "Service-road merge with courtesy and aggressive drivers",
                "Gap acceptance and ambiguity modelling",
                "Target speed adaptation"
            ],
        ),
        ScenarioDescriptor(
            name="Roundabout (simulated)",
            environment="OpenCV + Pygame",
            highlights=[
                "Four-way entry with multi-agent negotiation",
                "Yielding strategy stress tests",
                "Trajectory replanning under occlusions"
            ],
        ),
    ]

    notes = [
        "Dataset aggregates both empirical captures and procedural renders for comprehensive coverage.",
        "Labels support supervised learning as well as reinforcement learning reward shaping.",
        "Follow the paper's license and citation guidelines when redistributing derived datasets.",
    ]

    return DatasetMetadata(
        citation="Dixit, A., Kumar Chidambaram, R., & Allam, Z. (2021). Safety and Risk Analysis of Autonomous Vehicles Using Computer Vision and Neural Networks. Vehicles, 3(2), 595-617.",
        dataset_url="https://www.mdpi.com/2624-8921/3/2/32",
        modalities=modalities,
        annotations=annotations,
        scenarios=scenarios,
        notes=notes,
    )


def describe_dataset(print_fn=print) -> None:
    """Pretty-print the dataset description using the provided printer."""

    metadata = get_dataset_metadata()
    print_fn("Autonomous Vehicle Safety Dataset (Dixit et al., 2021)")
    print_fn(f"Citation: {metadata.citation}")
    print_fn(f"Reference URL: {metadata.dataset_url}\n")

    print_fn("Modalities:")
    for modality in metadata.modalities:
        print_fn(f"  - {modality.name} ({modality.source}) → {modality.format}")
        print_fn(f"    {modality.description}")

    print_fn("\nAnnotations:")
    for annotation in metadata.annotations:
        print_fn(f"  - {annotation.label} [{annotation.task}] — {annotation.description}")

    print_fn("\nScenarios:")
    for scenario in metadata.scenarios:
        print_fn(f"  - {scenario.name} ({scenario.environment})")
        for highlight in scenario.highlights:
            print_fn(f"      • {highlight}")

    print_fn("\nNotes:")
    for note in metadata.notes:
        print_fn(f"  - {note}")

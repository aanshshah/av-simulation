# Dataset Overview

The dataset referenced throughout this project is derived from the paper **"Safety and Risk Analysis of Autonomous Vehicles Using Computer Vision and Neural Networks"** by **Aditya Dixit, Rahul Kumar Chidambaram, and Zachary Allam (2021)**. It combines empirical sensor captures with procedurally generated episodes so that we can evaluate perception and planning pipelines under both realistic and repeatable conditions.

## Real-World Sensor Captures

- **Camera Footage** – Forward-facing RGB video recorded at 30 FPS in 1080p resolution. The sequences cover daytime, nighttime, and adverse weather (rain, fog) to stress-test lane detection and obstacle avoidance modules.
- **LiDAR Point Clouds** – Velodyne HDL-32E scans time-aligned with the video stream. They provide precise 3D localisation of vehicles, pedestrians, static obstacles, and lane boundaries.
- **Scene Labels** – Frame-level metadata distinguishes between straight highways, curved roads, intersections, and low-visibility conditions. These tags are used for triaging high-risk segments during training.

The real-world portion of the dataset ships with detailed supervision:

| Annotation Type | Targets | Primary Tasks |
|-----------------|---------|----------------|
| Bounding boxes  | Cars, trucks, buses, motorcycles, pedestrians | Object detection, trajectory forecasting |
| Instance masks  | Lane markings, drivable space | Semantic segmentation, lane tracking |
| Attributes      | Weather condition, lighting state, road curvature | Domain adaptation, risk-aware planning |

These labels allow us to train models that learn which actors or features (e.g., vulnerable road users, nearby vehicles, lane edges) are most predictive of collision risk.

## Simulated Scenarios

To complement the empirical data, the paper introduces synthetic episodes rendered with **OpenCV** and **Pygame**. The simulation provides deterministic control over traffic density, agent behaviour, and scene layout. Three canonical environments are recreated in this repository:

1. **Highway** – Four-lane freeway with configurable headway targets, speed limits, and sudden-brake hazards.
2. **Lane Merge** – Service-road merge that mixes courteous and aggressive drivers to evaluate gap acceptance.
3. **Roundabout** – Multi-entry roundabout that exercises yielding logic and manoeuvre negotiation under occlusions.

Each simulated episode exports RGB frames, occupancy masks, and action traces so that classical vision pipelines and reinforcement-learning agents can be validated before deployment on the real-world captures.

## Accessing Metadata Programmatically

The module `src/av_simulation/data/dataset_info.py` exposes `get_dataset_metadata()` and `describe_dataset()` helpers that summarise modalities, annotations, and scenarios. Example usage:

```python
from av_simulation.data import describe_dataset

describe_dataset()  # Prints a structured overview of sensors, labels, and benchmark scenes
```

## Citation

If you use this dataset or the accompanying simulation in your research, please cite the original paper:

> Dixit, A., Kumar Chidambaram, R., & Allam, Z. (2021). Safety and Risk Analysis of Autonomous Vehicles Using Computer Vision and Neural Networks. *Vehicles, 3*(2), 595–617. doi:10.3390/vehicles3020032

## Redistribution Notice

The dataset is published for research and educational purposes. When sharing derivatives, include the citation above and retain author attributions for both the empirical captures and the simulated assets.

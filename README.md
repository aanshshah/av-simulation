# Autonomous Vehicle Simulation Environment

Recreation of the simulation environment from the paper: **"Safety and Risk Analysis of Autonomous Vehicles Using Computer Vision and Neural Networks"** by Dixit et al. (2021)

## Overview

This project recreates the three main case studies from the paper:

1. **Straight Lane Detection** using Hough Line Transform
2. **Curved Lane Detection** using OpenCV and HSV color space
3. **Behavioral Planning** with Model-based Reinforcement Learning and Robust Control

## Project Structure

```
├── av_simulation.py          # Main simulation with 3 environments (highway, merge, roundabout)
├── lane_detection.py          # Lane detection algorithms (straight & curved)
├── behavioral_planning.py     # MDP, MRL, and robust control implementation
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Features

### 1. Simulation Environments (av_simulation.py)

Based on Case Study 3, implements three driving scenarios:

- **Highway Environment**: 4-lane highway with multiple vehicles
- **Lane Merging**: Highway with service road merge point
- **Roundabout**: 4-way roundabout navigation

**Key Parameters from Paper:**
- Acceleration Range: (-5, 5.0) m/s²
- Steering Range: ±45 degrees
- Max Speed: 40 m/s
- Default Speeds: [23, 25] m/s
- Perception Distance: 180 m

### 2. Lane Detection (lane_detection.py)

**Straight Lane Detection (Case Study 1):**
- Canny edge detection with 5×5 Gaussian filter
- Region of Interest (ROI) segmentation
- Hough Line Transform for lane marking detection
- Line averaging and extrapolation

**Curved Lane Detection (Case Study 2):**
- Camera distortion correction
- Perspective transformation (bird's eye view)
- HSV color space filtering for yellow/white lanes
- Sobel operator for edge detection
- 2nd-degree polynomial fitting: x = Ay² + By + C
- Radius of curvature calculation

### 3. Behavioral Planning (behavioral_planning.py)

**Model-Based Reinforcement Learning:**
- Neural network dynamics model: ẋ = f_θ(x, u) = A_θ(x, u)x + B_θ(x, u)u
- Experience buffer with 2000 training epochs
- Trajectory prediction with learned dynamics

**Planning Algorithms:**
- Cross-Entropy Method (CEM) for trajectory optimization
- Robust Control Framework with model uncertainty
- Continuous Ambiguity Prediction for neighboring vehicles
- Partially Observable MDP (POMDP) implementation

## Installation

### Google Colab
- Open the notebooks in `examples/notebooks/` and run `01_colab_setup.ipynb`.
- The setup notebook installs a minimal dependency set, downloads the source bundle, and provides links to the runner/analysis notebooks.

### Local (Replicable Environment)
1. Install Python 3.10 or 3.11. On Apple Silicon, Miniforge/conda-forge builds are recommended.
2. Create an isolated environment (choose one):
   ```bash
   # Option A: python -m venv
   python3 -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate

   # Option B: uv (reproducible sync)
   uv venv
   source .venv/bin/activate
   ```
3. Upgrade pip/uv and install the pinned dependency set:
   ```bash
   pip install --upgrade pip
   pip install -r requirements-local.txt
   ```
   > `requirements-local.txt` mirrors the Colab environment but pins versions so Macs, Windows, and Linux hosts share the same builds (pygame, OpenCV, torch, etc.).
4. Install the library in editable mode so the notebooks/scripts see the local source:
   ```bash
   pip install -e .
   ```
5. (Optional) Verify the install:
   ```bash
   python -m av_simulation.core.simulation --help  # quick import smoke test
   pytest tests                                  # if pytest is available
   ```

## Usage

### Run Main Simulation

```bash
python av_simulation.py
```

**Controls:**
- `1`, `2`, `3` - Switch between Highway, Merge, and Roundabout environments
- `SPACE` - Pause/Resume
- `R` - Reset current environment
- `Arrow Keys` - Manual vehicle control (for testing)
- `ESC` - Exit

The simulation includes:
- Green ego vehicle with autonomous behavior planning
- Blue traffic vehicles
- Collision detection
- Real-time speed and position display

### Run Lane Detection Demo

```bash
python lane_detection.py
```

**Controls:**
- `s` - Switch to Straight Lane Detection
- `c` - Switch to Curved Lane Detection
- `q` - Quit

The demo creates simulated road scenes and applies the detection algorithms.

### Run Behavioral Planning Demo

```bash
python behavioral_planning.py
```

This runs a demonstration of:
1. Model training with synthetic data
2. Trajectory prediction
3. Cross-entropy optimization
4. Robust control with uncertainty
5. Continuous ambiguity prediction

## Implementation Details

### Vehicle Dynamics

The vehicle model uses bicycle dynamics with state vector:
- Position (x, y)
- Velocity (vx, vy)
- Heading angle
- Current lane

### Behavioral Planner

The planner considers:
- Distance to front vehicle
- Safe lane change opportunities
- Speed maintenance within limits
- Collision avoidance

### Robust Control

Addresses model errors from Case Study 3:
- Considers multiple possible future states
- Worst-case scenario planning
- Driving style estimation (aggressive/normal/conservative)

## Key Findings Recreated

1. **Lane Detection**: Successfully detects both straight and curved lanes using computer vision
2. **Behavioral Planning**: MDP-based planning reduces collision risk
3. **Robust Control**: Considering uncertainty prevents accidents shown in Figure 11 of the paper

## Simulation Parameters

All parameters match Table 2 from the paper:

| Parameter | Value |
|-----------|-------|
| Acceleration Range | (-5, 5.0) m/s² |
| Steering Range | (-0.785, 0.785) rad |
| Max Speed | 40 m/s |
| Default Speeds | [23, 25] m/s |
| Distance Wanted | 10.0 m |
| Time Wanted | 1.5 s |
| Perception Distance | 180 m |

## Limitations

This is a simplified recreation for educational purposes:
- Uses pygame instead of actual vehicle hardware
- Synthetic data instead of real LIDAR/camera feeds
- Simplified physics model
- No actual V2X communication

## Future Improvements

- Add pedestrian and cyclist models
- Implement V2X communication simulation
- Add more complex road scenarios
- Include weather conditions (fog, rain)
- Implement full SCNN for better spatial feature detection

## References

Dixit, A., Kumar Chidambaram, R., & Allam, Z. (2021). Safety and Risk Analysis of Autonomous Vehicles Using Computer Vision and Neural Networks. Vehicles, 3, 595-617.

## License

This is an educational recreation of academic research. Please refer to the original paper for scientific details.

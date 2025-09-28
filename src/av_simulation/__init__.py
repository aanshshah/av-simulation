"""
Autonomous Vehicle Simulation Package

A comprehensive simulation environment for autonomous vehicle research
including lane detection, behavioral planning, and various driving scenarios.
"""

__version__ = "1.0.0"
__author__ = "Aansh Shah"

# Import core simulation components
try:
    from .core.simulation import *
except ImportError as e:
    print(f"Warning: Could not import simulation core: {e}")

# Import detection components (optional)
try:
    from .detection.lane_detection import *
except ImportError as e:
    print(f"Warning: Could not import lane detection: {e}")

# Import planning components (optional)
try:
    from .planning.behavioral_planning import *
except ImportError as e:
    print(f"Warning: Could not import behavioral planning: {e}")

__all__ = [
    "SimulationEnvironment",
    "Vehicle",
    "LaneDetector",
    "BehavioralPlanner"
]
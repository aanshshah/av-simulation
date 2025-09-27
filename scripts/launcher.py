#!/usr/bin/env python3
"""
Launcher for Autonomous Vehicle Simulation
Quick access to all simulation modules
"""

import sys
import subprocess
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def print_header():
    print("=" * 60)
    print("  AUTONOMOUS VEHICLE SIMULATION LAUNCHER")
    print("  Based on: Safety and Risk Analysis of AVs")
    print("  Dixit et al. (2021)")
    print("=" * 60)

def print_menu():
    print("\nSelect a simulation to run:\n")
    print("  1. Main AV Simulation (Highway/Merge/Roundabout)")
    print("  2. Lane Detection Demo (Straight & Curved)")
    print("  3. Behavioral Planning Demo (MDP/Robust Control)")
    print("  4. Install Requirements")
    print("  5. View README")
    print("  0. Exit")
    print()

def run_simulation():
    """Run the main vehicle simulation"""
    print("\nLaunching Main AV Simulation...")
    print("Controls: 1/2/3 to switch environments, SPACE to pause, R to reset")
    print("-" * 40)
    subprocess.run([sys.executable, "-m", "av_simulation.core.simulation"])

def run_lane_detection():
    """Run lane detection demo"""
    print("\nLaunching Lane Detection Demo...")
    print("Controls: 's' for straight lanes, 'c' for curved lanes")
    print("-" * 40)
    subprocess.run([sys.executable, "-m", "av_simulation.detection.lane_detection"])

def run_behavioral_planning():
    """Run behavioral planning demo"""
    print("\nLaunching Behavioral Planning Demo...")
    print("This will demonstrate MDP, MRL, and Robust Control")
    print("-" * 40)
    subprocess.run([sys.executable, "-m", "av_simulation.planning.behavioral_planning"])

def install_requirements():
    """Install required packages"""
    print("\nInstalling requirements...")
    print("-" * 40)
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("\nInstallation complete!")

def view_readme():
    """Display README content"""
    try:
        with open("README.md", "r") as f:
            content = f.read()
        print("\n" + "=" * 60)
        print(content)
        print("=" * 60)
        input("\nPress Enter to continue...")
    except FileNotFoundError:
        print("README.md not found!")

def main():
    while True:
        print_header()
        print_menu()
        
        try:
            choice = input("Enter your choice (0-5): ").strip()
            
            if choice == "1":
                run_simulation()
            elif choice == "2":
                run_lane_detection()
            elif choice == "3":
                run_behavioral_planning()
            elif choice == "4":
                install_requirements()
            elif choice == "5":
                view_readme()
            elif choice == "0":
                print("\nExiting... Thank you for using AV Simulation!")
                break
            else:
                print("\nInvalid choice! Please enter 0-5.")
                input("Press Enter to continue...")
        except KeyboardInterrupt:
            print("\n\nInterrupted. Exiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()

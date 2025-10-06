"""
Helper functions for running AV simulation in Google Colab environment.

This module provides enhanced compatibility with Google Colab and includes
robust error handling, dynamic path detection, and memory management.
"""

import os
import sys
import time
import threading
from typing import Optional, Dict, List, Any, Tuple
from PIL import Image
import matplotlib.pyplot as plt
import pygame
import numpy as np

try:
    from .colab_setup_utils import (
        ColabEnvironmentDetector,
        ColabDependencyManager,
        ColabDisplayManager as BaseColabDisplayManager,
        ColabProjectManager
    )
except ImportError:
    # Fallback for when colab_setup_utils is not available
    print("⚠️ colab_setup_utils not available, using basic functionality")
    ColabEnvironmentDetector = None
    ColabDependencyManager = None
    BaseColabDisplayManager = None
    ColabProjectManager = None


class ColabDisplayManager:
    """Manages virtual display setup for Colab"""

    def __init__(self):
        self.display = None
        self.is_setup = False

    def setup_virtual_display(self, width=1200, height=800):
        """Setup virtual display for pygame in Colab"""
        try:
            from pyvirtualdisplay import Display

            self.display = Display(visible=0, size=(width, height))
            self.display.start()

            os.environ['DISPLAY'] = ':' + str(self.display.display)
            self.is_setup = True

            print(f"✅ Virtual display setup: {width}x{height}")
            print(f"🖥️  Display ID: {os.environ.get('DISPLAY')}")

            return True

        except ImportError:
            print("❌ pyvirtualdisplay not available")
            return False
        except Exception as e:
            print(f"❌ Display setup failed: {e}")
            return False

    def test_pygame(self):
        """Test pygame functionality"""
        if not self.is_setup:
            print("❌ Display not setup")
            return False

        try:
            pygame.init()
            screen = pygame.display.set_mode((400, 300))
            pygame.display.set_caption("Test")

            # Draw test pattern
            screen.fill((100, 100, 100))
            pygame.draw.circle(screen, (255, 0, 0), (200, 150), 50)
            pygame.display.flip()

            pygame.quit()
            print("✅ Pygame test successful")
            return True

        except Exception as e:
            print(f"❌ Pygame test failed: {e}")
            return False

    def cleanup(self):
        """Cleanup virtual display"""
        if self.display:
            self.display.stop()
            self.is_setup = False
            print("🧹 Virtual display cleaned up")


class ColabSimulationRunner:
    """Enhanced simulation runner for Colab environment"""

    def __init__(self, display_manager=None, max_screenshots: int = 50):
        self.display_manager = display_manager or ColabDisplayManager()
        self.screenshots = []
        self.max_screenshots = max_screenshots
        self.simulation_data = {}
        self.running = False
        self.env_detector = ColabEnvironmentDetector() if ColabEnvironmentDetector else None

    def setup_environment(self):
        """Setup complete Colab environment"""
        print("🔧 Setting up Colab environment...")

        # Setup display
        if not self.display_manager.setup_virtual_display():
            return False

        # Test pygame
        if not self.display_manager.test_pygame():
            return False

        # Setup paths dynamically
        if ColabEnvironmentDetector:
            detector = ColabEnvironmentDetector()
            sim_path = detector.get_simulation_path()
            if sim_path not in sys.path:
                sys.path.insert(0, sim_path)
                print(f"📁 Added to Python path: {sim_path}")
        else:
            # Fallback to original behavior
            if '/content' in os.getcwd():
                sys.path.insert(0, '/content/av-simulation/src')

        print("✅ Colab environment ready")
        return True

    def capture_screenshot(self, screen, metadata=None):
        """Capture and store screenshot with metadata"""
        w, h = screen.get_size()
        raw = pygame.image.tostring(screen, 'RGB')
        image = Image.frombytes('RGB', (w, h), raw)

        screenshot_data = {
            'image': image,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }

        # Implement circular buffer for memory management
        if len(self.screenshots) >= self.max_screenshots:
            self.screenshots.pop(0)  # Remove oldest screenshot
            print(f"🗑️ Removed old screenshot (keeping last {self.max_screenshots})")

        self.screenshots.append(screenshot_data)
        return screenshot_data

    def run_headless_simulation(self, config):
        """Run simulation without GUI for data collection"""
        print(f"🚀 Starting headless simulation: {config}")

        # Import here to avoid issues
        try:
            from av_simulation.core.simulation import Simulation
            print("✅ Simulation module imported successfully")
        except ImportError as e:
            print(f"❌ Import failed: {e}")
            print("💡 Suggestions:")
            print("  1. Run the setup cells in 01_colab_setup.ipynb first")
            print("  2. Upload simulation files manually")
            print("  3. Clone from GitHub if repository is public")
            print("  4. Check that av_simulation package is in Python path")
            return None

        # Create simulation
        sim = Simulation(enable_data_collection=config.get('collect_data', True))

        # Configure environment
        env_type = config.get('environment', 'highway')
        if env_type in sim.environments:
            sim.switch_environment(env_type)

        # Run simulation
        duration = config.get('duration', 30)
        screenshot_interval = config.get('screenshot_interval', 2.0)

        start_time = time.time()
        last_screenshot = 0

        try:
            while time.time() - start_time < duration:
                current_time = time.time() - start_time

                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        break

                # Update simulation
                if sim.current_env.ego_vehicle:
                    action = sim.planner.plan_action(sim.current_env.ego_vehicle)
                    sim.current_env.ego_vehicle.set_action(action)

                sim.current_env.step(1.0/60)

                # Collect data
                if sim.data_collection_enabled:
                    sim.collect_simulation_data()

                # Capture screenshots
                if current_time - last_screenshot >= screenshot_interval:
                    sim.current_env.draw(sim.screen)
                    sim.draw_hud()
                    pygame.display.flip()

                    metadata = {
                        'simulation_time': current_time,
                        'environment': env_type,
                        'ego_speed': sim.current_env.ego_vehicle.state.vx if sim.current_env.ego_vehicle else 0,
                        'collision': sim.current_env.collision_detected
                    }

                    self.capture_screenshot(sim.screen, metadata)
                    last_screenshot = current_time

                    print(f"📸 Screenshot at t={current_time:.1f}s")

                time.sleep(1/60)

            # Save final data
            if sim.data_collection_enabled and sim.current_run_id:
                sim.data_repository.end_current_run()
                print(f"💾 Data saved: {sim.current_run_id}")

            pygame.quit()

            return {
                'run_id': sim.current_run_id if sim.data_collection_enabled else None,
                'repository': sim.data_repository if sim.data_collection_enabled else None,
                'screenshots': len(self.screenshots),
                'duration': time.time() - start_time
            }

        except Exception as e:
            print(f"❌ Simulation error: {e}")
            pygame.quit()
            return None

    def display_screenshots(self, max_images=6, figsize=(15, 10)):
        """Display captured screenshots in a grid"""
        if not self.screenshots:
            print("No screenshots to display")
            return

        n_images = min(len(self.screenshots), max_images)
        cols = 3
        rows = (n_images + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()

        for i in range(n_images):
            screenshot = self.screenshots[i]
            axes[i].imshow(screenshot['image'])

            # Create title with metadata
            metadata = screenshot['metadata']
            title = f"t={metadata.get('simulation_time', 0):.1f}s"
            if metadata.get('collision', False):
                title += " (COLLISION!)"
                axes[i].set_facecolor('red')

            axes[i].set_title(title)
            axes[i].axis('off')

        # Hide empty subplots
        for i in range(n_images, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    def export_screenshots(self, output_dir="screenshots"):
        """Export screenshots to files"""
        if not self.screenshots:
            print("No screenshots to export")
            return

        os.makedirs(output_dir, exist_ok=True)

        for i, screenshot in enumerate(self.screenshots):
            filename = f"screenshot_{i:03d}.png"
            filepath = os.path.join(output_dir, filename)
            screenshot['image'].save(filepath)

        print(f"📁 Exported {len(self.screenshots)} screenshots to {output_dir}")
        return output_dir

    def cleanup(self):
        """Cleanup resources"""
        if self.display_manager:
            self.display_manager.cleanup()

        self.screenshots.clear()
        print("🧹 Simulation runner cleaned up")


def install_colab_dependencies():
    """Install required packages for Colab"""
    packages = [
        "pygame",
        "pyvirtualdisplay",
        "pandas",
        "matplotlib",
        "seaborn",
        "plotly",
        "scipy",
        "scikit-learn",
        "pillow"
    ]

    print("📦 Installing Colab dependencies...")

    import subprocess
    import sys

    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package}")
        except subprocess.CalledProcessError:
            print(f"❌ {package}")

    print("📦 Installation complete")


def setup_colab_environment():
    """Complete Colab environment setup"""
    print("🔧 Setting up complete Colab environment...")

    # Install system dependencies
    import subprocess
    subprocess.run(["apt-get", "update"], capture_output=True)
    subprocess.run(["apt-get", "install", "-y", "xvfb", "python3-opengl"], capture_output=True)

    # Install Python packages
    install_colab_dependencies()

    # Setup display manager
    display_manager = ColabDisplayManager()
    if not display_manager.setup_virtual_display():
        return None

    # Test setup
    if not display_manager.test_pygame():
        return None

    print("✅ Colab environment fully configured")
    return display_manager


def download_simulation_code(repo_url="https://github.com/aanshshah/av-simulation.git"):
    """Download simulation code from repository"""
    print(f"📥 Downloading simulation code from {repo_url}")

    import subprocess

    try:
        # Clone repository
        result = subprocess.run(["git", "clone", repo_url],
                              capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Code downloaded successfully")
            return True
        else:
            print(f"❌ Git clone failed: {result.stderr}")
            return False

    except FileNotFoundError:
        print("❌ Git not available")
        return False
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


# Convenience functions for quick setup
def quick_colab_setup(download_code: bool = False, repo_url: str = "https://github.com/aanshshah/av-simulation.git"):
    """Enhanced one-command Colab setup with better error handling"""
    print("🚀 Quick Colab Setup for AV Simulation")
    print("=" * 50)

    # Use enhanced setup if available
    if ColabEnvironmentDetector:
        try:
            from .colab_setup_utils import ColabSetupCoordinator
            coordinator = ColabSetupCoordinator()
            results = coordinator.run_complete_setup(download_code=download_code, repo_url=repo_url)

            if results.get('display_setup', False):
                runner = ColabSimulationRunner(coordinator.display_manager)
                print("\n✅ Enhanced setup complete!")
                print("\n📋 Next steps:")
                print("1. Use runner.run_headless_simulation(config) to run simulations")
                print("2. Use runner.display_screenshots() to view results")
                return runner
            else:
                print("❌ Setup failed - check error messages above")
                return None
        except Exception as e:
            print(f"⚠️ Enhanced setup failed, falling back to basic setup: {e}")

    # Fallback to original setup
    print("\n🔄 Using basic setup...")
    display_manager = setup_colab_environment()
    if not display_manager:
        print("❌ Environment setup failed")
        return None

    runner = ColabSimulationRunner(display_manager)

    print("✅ Basic setup complete!")
    print("\n📋 Next steps:")
    print("1. Upload your simulation files or clone from GitHub")
    print("2. Use runner.run_headless_simulation(config) to run simulations")
    print("3. Use runner.display_screenshots() to view results")

    return runner


if __name__ == "__main__":
    # Test the setup
    runner = quick_colab_setup()
    if runner:
        print("\n🎉 Colab helpers ready to use!")
    else:
        print("\n❌ Setup failed - check error messages above")
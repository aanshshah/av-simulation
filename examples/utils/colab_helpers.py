"""
Helper functions for running AV simulation in Google Colab environment.
"""

import os
import sys
import time
import threading
from PIL import Image
import matplotlib.pyplot as plt
import pygame


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

    def __init__(self, display_manager=None):
        self.display_manager = display_manager or ColabDisplayManager()
        self.screenshots = []
        self.simulation_data = {}
        self.running = False

    def setup_environment(self):
        """Setup complete Colab environment"""
        print("🔧 Setting up Colab environment...")

        # Setup display
        if not self.display_manager.setup_virtual_display():
            return False

        # Test pygame
        if not self.display_manager.test_pygame():
            return False

        # Setup paths
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

        self.screenshots.append(screenshot_data)
        return screenshot_data

    def run_headless_simulation(self, config):
        """Run simulation without GUI for data collection"""
        print(f"🚀 Starting headless simulation: {config}")

        # Import here to avoid issues
        try:
            from av_simulation.core.simulation import Simulation
        except ImportError as e:
            print(f"❌ Import failed: {e}")
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
                    }\n                    \n                    self.capture_screenshot(sim.screen, metadata)\n                    last_screenshot = current_time\n                    \n                    print(f\"📸 Screenshot at t={current_time:.1f}s\")\n                \n                time.sleep(1/60)\n            \n            # Save final data\n            if sim.data_collection_enabled and sim.current_run_id:\n                sim.data_repository.end_current_run()\n                print(f\"💾 Data saved: {sim.current_run_id}\")\n            \n            pygame.quit()\n            \n            return {\n                'run_id': sim.current_run_id if sim.data_collection_enabled else None,\n                'repository': sim.data_repository if sim.data_collection_enabled else None,\n                'screenshots': len(self.screenshots),\n                'duration': time.time() - start_time\n            }\n            \n        except Exception as e:\n            print(f\"❌ Simulation error: {e}\")\n            pygame.quit()\n            return None\n    \n    def display_screenshots(self, max_images=6, figsize=(15, 10)):\n        \"\"\"Display captured screenshots in a grid\"\"\"\n        if not self.screenshots:\n            print(\"No screenshots to display\")\n            return\n        \n        n_images = min(len(self.screenshots), max_images)\n        cols = 3\n        rows = (n_images + cols - 1) // cols\n        \n        fig, axes = plt.subplots(rows, cols, figsize=figsize)\n        if rows == 1:\n            axes = [axes] if cols == 1 else axes\n        else:\n            axes = axes.flatten()\n        \n        for i in range(n_images):\n            screenshot = self.screenshots[i]\n            axes[i].imshow(screenshot['image'])\n            \n            # Create title with metadata\n            metadata = screenshot['metadata']\n            title = f\"t={metadata.get('simulation_time', 0):.1f}s\"\n            if metadata.get('collision', False):\n                title += \" (COLLISION!)\"\n                axes[i].set_facecolor('red')\n            \n            axes[i].set_title(title)\n            axes[i].axis('off')\n        \n        # Hide empty subplots\n        for i in range(n_images, len(axes)):\n            axes[i].axis('off')\n        \n        plt.tight_layout()\n        plt.show()\n    \n    def export_screenshots(self, output_dir=\"screenshots\"):\n        \"\"\"Export screenshots to files\"\"\"\n        if not self.screenshots:\n            print(\"No screenshots to export\")\n            return\n        \n        os.makedirs(output_dir, exist_ok=True)\n        \n        for i, screenshot in enumerate(self.screenshots):\n            filename = f\"screenshot_{i:03d}.png\"\n            filepath = os.path.join(output_dir, filename)\n            screenshot['image'].save(filepath)\n        \n        print(f\"📁 Exported {len(self.screenshots)} screenshots to {output_dir}\")\n        return output_dir\n    \n    def cleanup(self):\n        \"\"\"Cleanup resources\"\"\"\n        if self.display_manager:\n            self.display_manager.cleanup()\n        \n        self.screenshots.clear()\n        print(\"🧹 Simulation runner cleaned up\")\n\n\ndef install_colab_dependencies():\n    \"\"\"Install required packages for Colab\"\"\"\n    packages = [\n        \"pygame\",\n        \"pyvirtualdisplay\",\n        \"pandas\",\n        \"matplotlib\",\n        \"seaborn\",\n        \"plotly\",\n        \"scipy\",\n        \"scikit-learn\",\n        \"pillow\"\n    ]\n    \n    print(\"📦 Installing Colab dependencies...\")\n    \n    import subprocess\n    import sys\n    \n    for package in packages:\n        try:\n            subprocess.check_call([sys.executable, \"-m\", \"pip\", \"install\", package])\n            print(f\"✅ {package}\")\n        except subprocess.CalledProcessError:\n            print(f\"❌ {package}\")\n    \n    print(\"📦 Installation complete\")\n\n\ndef setup_colab_environment():\n    \"\"\"Complete Colab environment setup\"\"\"\n    print(\"🔧 Setting up complete Colab environment...\")\n    \n    # Install system dependencies\n    import subprocess\n    subprocess.run([\"apt-get\", \"update\"], capture_output=True)\n    subprocess.run([\"apt-get\", \"install\", \"-y\", \"xvfb\", \"python3-opengl\"], capture_output=True)\n    \n    # Install Python packages\n    install_colab_dependencies()\n    \n    # Setup display manager\n    display_manager = ColabDisplayManager()\n    if not display_manager.setup_virtual_display():\n        return None\n    \n    # Test setup\n    if not display_manager.test_pygame():\n        return None\n    \n    print(\"✅ Colab environment fully configured\")\n    return display_manager\n\n\ndef download_simulation_code(repo_url=\"https://github.com/aanshshah/av-simulation.git\"):\n    \"\"\"Download simulation code from repository\"\"\"\n    print(f\"📥 Downloading simulation code from {repo_url}\")\n    \n    import subprocess\n    \n    try:\n        # Clone repository\n        result = subprocess.run([\"git\", \"clone\", repo_url], \n                              capture_output=True, text=True)\n        \n        if result.returncode == 0:\n            print(\"✅ Code downloaded successfully\")\n            return True\n        else:\n            print(f\"❌ Git clone failed: {result.stderr}\")\n            return False\n            \n    except FileNotFoundError:\n        print(\"❌ Git not available\")\n        return False\n    except Exception as e:\n        print(f\"❌ Download failed: {e}\")\n        return False\n\n\n# Convenience functions for quick setup\ndef quick_colab_setup():\n    \"\"\"One-command Colab setup\"\"\"\n    print(\"🚀 Quick Colab Setup for AV Simulation\")\n    print(\"=\" * 50)\n    \n    # Setup environment\n    display_manager = setup_colab_environment()\n    if not display_manager:\n        print(\"❌ Environment setup failed\")\n        return None\n    \n    # Create runner\n    runner = ColabSimulationRunner(display_manager)\n    \n    print(\"✅ Quick setup complete!\")\n    print(\"\\n📋 Next steps:\")\n    print(\"1. Upload your simulation files or clone from GitHub\")\n    print(\"2. Use runner.run_headless_simulation(config) to run simulations\")\n    print(\"3. Use runner.display_screenshots() to view results\")\n    \n    return runner\n\n\nif __name__ == \"__main__\":\n    # Test the setup\n    runner = quick_colab_setup()\n    if runner:\n        print(\"\\n🎉 Colab helpers ready to use!\")\n    else:\n        print(\"\\n❌ Setup failed - check error messages above\")
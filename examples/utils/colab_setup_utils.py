"""
Google Colab Setup Utilities for AV Simulation

This module provides robust utilities for setting up and running
the AV simulation in Google Colab environments.
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


class ColabEnvironmentDetector:
    """Detects and provides information about the current environment."""

    @staticmethod
    def is_colab() -> bool:
        """Check if running in Google Colab."""
        return (
            'COLAB_GPU' in os.environ or
            'COLAB_TPU_ADDR' in os.environ or
            '/content' in os.getcwd() or
            'google.colab' in sys.modules
        )

    @staticmethod
    def get_base_path() -> str:
        """Get the appropriate base path for the current environment."""
        if ColabEnvironmentDetector.is_colab():
            return '/content'
        else:
            return os.getcwd()

    @staticmethod
    def get_simulation_path() -> str:
        """Get the path where simulation code should be located."""
        base = ColabEnvironmentDetector.get_base_path()
        return os.path.join(base, 'av-simulation', 'src')

    @staticmethod
    def get_data_path() -> str:
        """Get the path for simulation data storage."""
        base = ColabEnvironmentDetector.get_base_path()
        return os.path.join(base, 'simulation_data')

    @staticmethod
    def get_environment_info() -> Dict[str, Any]:
        """Get comprehensive environment information."""
        info = {
            'is_colab': ColabEnvironmentDetector.is_colab(),
            'base_path': ColabEnvironmentDetector.get_base_path(),
            'simulation_path': ColabEnvironmentDetector.get_simulation_path(),
            'data_path': ColabEnvironmentDetector.get_data_path(),
            'python_version': sys.version,
            'current_dir': os.getcwd(),
            'env_vars': {
                'DISPLAY': os.environ.get('DISPLAY'),
                'COLAB_GPU': os.environ.get('COLAB_GPU'),
                'COLAB_TPU_ADDR': os.environ.get('COLAB_TPU_ADDR')
            }
        }

        # Add hardware info if available
        try:
            import psutil
            info['memory_gb'] = round(psutil.virtual_memory().total / (1024**3), 1)
            info['cpu_count'] = psutil.cpu_count()
        except ImportError:
            info['memory_gb'] = 'unknown'
            info['cpu_count'] = 'unknown'

        # Check GPU availability
        try:
            import torch
            info['gpu_available'] = torch.cuda.is_available()
            if info['gpu_available']:
                info['gpu_name'] = torch.cuda.get_device_name(0)
        except ImportError:
            try:
                import tensorflow as tf
                info['gpu_available'] = len(tf.config.experimental.list_physical_devices('GPU')) > 0
            except ImportError:
                info['gpu_available'] = 'unknown'

        return info


class ColabDependencyManager:
    """Manages dependencies and package installation for Colab."""

    SYSTEM_PACKAGES = [
        'xvfb',
        'python3-opengl',
        'ffmpeg',
        'git'
    ]

    PYTHON_PACKAGES = [
        'pygame',
        'pyvirtualdisplay',
        'pillow',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'plotly',
        'scipy',
        'scikit-learn'
    ]

    @staticmethod
    def install_system_packages(packages: Optional[List[str]] = None, quiet: bool = True) -> bool:
        """Install system packages using apt-get."""
        if packages is None:
            packages = ColabDependencyManager.SYSTEM_PACKAGES

        print("📦 Installing system dependencies...")

        try:
            # Update package list
            cmd = ["apt-get", "update"]
            if quiet:
                cmd.extend(["-qq"])

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Failed to update package list: {result.stderr}")
                return False

            # Install packages
            cmd = ["apt-get", "install", "-y"] + packages
            if quiet:
                cmd.extend(["> /dev/null", "2>&1"])

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Failed to install system packages: {result.stderr}")
                return False

            print("✅ System dependencies installed")
            return True

        except Exception as e:
            print(f"❌ System package installation error: {e}")
            return False

    @staticmethod
    def install_python_packages(packages: Optional[List[str]] = None, quiet: bool = True) -> bool:
        """Install Python packages using pip."""
        if packages is None:
            packages = ColabDependencyManager.PYTHON_PACKAGES

        print("🐍 Installing Python packages...")

        success_count = 0
        for package in packages:
            try:
                cmd = [sys.executable, "-m", "pip", "install", package]
                if quiet:
                    cmd.extend(["--quiet"])

                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"  ✅ {package}")
                    success_count += 1
                else:
                    print(f"  ❌ {package}: {result.stderr.strip()}")

            except Exception as e:
                print(f"  ❌ {package}: {e}")

        print(f"📦 Python packages: {success_count}/{len(packages)} installed")
        return success_count == len(packages)

    @staticmethod
    def verify_installation() -> Dict[str, bool]:
        """Verify that all required packages are available."""
        verification = {}

        # Check Python packages
        python_packages = {
            'pygame': 'pygame',
            'pyvirtualdisplay': 'pyvirtualdisplay',
            'PIL': 'pillow',
            'pandas': 'pandas',
            'numpy': 'numpy',
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn',
            'plotly': 'plotly',
            'scipy': 'scipy',
            'sklearn': 'scikit-learn'
        }

        for module, package in python_packages.items():
            try:
                __import__(module)
                verification[package] = True
            except ImportError:
                verification[package] = False

        return verification


class ColabDisplayManager:
    """Manages virtual display setup for Colab."""

    def __init__(self):
        self.display = None
        self.is_setup = False
        self.display_size = (1200, 800)

    def setup_display(self, width: int = 1200, height: int = 800, retries: int = 3) -> bool:
        """Setup virtual display with retry logic."""
        self.display_size = (width, height)

        for attempt in range(retries):
            if self._attempt_display_setup():
                return True
            else:
                print(f"⚠️ Display setup attempt {attempt + 1} failed, retrying...")
                time.sleep(1)

        print("❌ Display setup failed after all retries")
        return False

    def _attempt_display_setup(self) -> bool:
        """Single attempt to setup display."""
        try:
            from pyvirtualdisplay import Display

            # Stop existing display if any
            if self.display:
                try:
                    self.display.stop()
                except:
                    pass

            # Create new display
            self.display = Display(visible=0, size=self.display_size)
            self.display.start()

            # Set environment variable
            os.environ['DISPLAY'] = ':' + str(self.display.display)

            # Test the display
            if self._test_display():
                self.is_setup = True
                print(f"✅ Virtual display ready: {self.display_size[0]}x{self.display_size[1]}")
                print(f"🖥️  Display ID: {os.environ.get('DISPLAY')}")
                return True
            else:
                return False

        except ImportError:
            print("❌ pyvirtualdisplay not available - install dependencies first")
            return False
        except Exception as e:
            print(f"❌ Display setup failed: {e}")
            return False

    def _test_display(self) -> bool:
        """Test if display is working properly."""
        try:
            import pygame
            pygame.init()
            screen = pygame.display.set_mode((100, 100))
            pygame.display.flip()
            pygame.quit()
            return True
        except Exception as e:
            print(f"❌ Display test failed: {e}")
            return False

    def cleanup(self):
        """Cleanup display resources."""
        if self.display:
            try:
                self.display.stop()
                self.is_setup = False
                print("🧹 Display cleaned up")
            except:
                pass

    def get_status(self) -> Dict[str, Any]:
        """Get display status information."""
        return {
            'is_setup': self.is_setup,
            'display_size': self.display_size,
            'display_env': os.environ.get('DISPLAY'),
            'display_object': self.display is not None
        }


class ColabProjectManager:
    """Manages simulation project files and setup."""

    def __init__(self):
        self.env_detector = ColabEnvironmentDetector()
        self.base_path = self.env_detector.get_base_path()
        self.simulation_path = self.env_detector.get_simulation_path()

    def check_simulation_files(self) -> Dict[str, bool]:
        """Check if simulation files are available."""
        required_paths = {
            'simulation_src': os.path.join(self.base_path, 'av-simulation', 'src'),
            'av_simulation_package': os.path.join(self.base_path, 'av-simulation', 'src', 'av_simulation'),
            'core_module': os.path.join(self.base_path, 'av-simulation', 'src', 'av_simulation', 'core'),
            'data_module': os.path.join(self.base_path, 'av-simulation', 'src', 'av_simulation', 'data')
        }

        status = {}
        for name, path in required_paths.items():
            status[name] = os.path.exists(path)

        return status

    def setup_python_path(self) -> bool:
        """Add simulation source to Python path."""
        try:
            if self.simulation_path not in sys.path:
                sys.path.insert(0, self.simulation_path)
                print(f"✅ Added to Python path: {self.simulation_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to setup Python path: {e}")
            return False

    def download_from_github(self, repo_url: str = "https://github.com/aanshshah/av-simulation.git") -> bool:
        """Download simulation code from GitHub."""
        print(f"📥 Downloading from: {repo_url}")

        try:
            # Change to base directory
            original_dir = os.getcwd()
            os.chdir(self.base_path)

            # Clone repository
            result = subprocess.run(
                ["git", "clone", repo_url],
                capture_output=True,
                text=True
            )

            os.chdir(original_dir)

            if result.returncode == 0:
                print("✅ Repository cloned successfully")
                return self.setup_python_path()
            else:
                print(f"❌ Git clone failed: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False

    def create_data_directory(self) -> str:
        """Create and return data directory path."""
        data_path = self.env_detector.get_data_path()
        try:
            os.makedirs(data_path, exist_ok=True)
            print(f"📁 Data directory ready: {data_path}")
            return data_path
        except Exception as e:
            print(f"❌ Failed to create data directory: {e}")
            return ""

    def verify_imports(self) -> Dict[str, bool]:
        """Verify that simulation modules can be imported."""
        imports_to_test = {
            'av_simulation': 'av_simulation',
            'simulation_core': 'av_simulation.core.simulation',
            'data_repository': 'av_simulation.data.repository',
            'data_exporters': 'av_simulation.data.exporters'
        }

        results = {}
        for name, module_path in imports_to_test.items():
            try:
                __import__(module_path)
                results[name] = True
                print(f"✅ {name}: import successful")
            except ImportError as e:
                results[name] = False
                print(f"❌ {name}: {e}")

        return results


class ColabSetupCoordinator:
    """Coordinates the complete Colab setup process."""

    def __init__(self):
        self.env_detector = ColabEnvironmentDetector()
        self.dependency_manager = ColabDependencyManager()
        self.display_manager = ColabDisplayManager()
        self.project_manager = ColabProjectManager()

    def run_complete_setup(self,
                         install_dependencies: bool = True,
                         setup_display: bool = True,
                         download_code: bool = False,
                         repo_url: str = "https://github.com/aanshshah/av-simulation.git") -> Dict[str, bool]:
        """Run the complete setup process."""

        print("🚀 Starting Complete Colab Setup")
        print("=" * 50)

        results = {}

        # Environment detection
        env_info = self.env_detector.get_environment_info()
        print(f"🌍 Environment: {'Google Colab' if env_info['is_colab'] else 'Local'}")
        print(f"📁 Base path: {env_info['base_path']}")

        # Install dependencies
        if install_dependencies:
            print("\n1. Installing Dependencies...")
            results['system_packages'] = self.dependency_manager.install_system_packages()
            results['python_packages'] = self.dependency_manager.install_python_packages()

        # Setup display
        if setup_display:
            print("\n2. Setting up Virtual Display...")
            results['display_setup'] = self.display_manager.setup_display()

        # Download/setup project files
        print("\n3. Setting up Project Files...")
        file_status = self.project_manager.check_simulation_files()

        if not any(file_status.values()) and download_code:
            print("No simulation files found, downloading from GitHub...")
            results['code_download'] = self.project_manager.download_from_github(repo_url)
        else:
            print("Using existing simulation files or manual upload required")
            results['code_download'] = any(file_status.values())

        # Setup Python path
        results['python_path'] = self.project_manager.setup_python_path()

        # Create data directory
        data_path = self.project_manager.create_data_directory()
        results['data_directory'] = bool(data_path)

        # Verify setup
        print("\n4. Verifying Setup...")
        verification = self.dependency_manager.verify_installation()
        import_results = self.project_manager.verify_imports()

        results['verification'] = all(verification.values())
        results['imports'] = all(import_results.values())

        # Summary
        print("\n" + "=" * 50)
        print("📋 Setup Summary:")
        for step, success in results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {step.replace('_', ' ').title()}")

        overall_success = all(results.values())
        print(f"\n🎯 Overall: {'✅ SUCCESS' if overall_success else '❌ ISSUES FOUND'}")

        if not overall_success:
            print("\n💡 Next steps:")
            if not results.get('code_download', True):
                print("  - Upload simulation files manually or check GitHub URL")
            if not results.get('imports', True):
                print("  - Verify simulation code structure and dependencies")
            print("  - Check error messages above for specific issues")

        return results

    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report."""
        return {
            'environment': self.env_detector.get_environment_info(),
            'display': self.display_manager.get_status(),
            'files': self.project_manager.check_simulation_files(),
            'dependencies': self.dependency_manager.verify_installation()
        }


# Convenience function for quick setup
def quick_colab_setup(**kwargs) -> ColabSetupCoordinator:
    """Quick one-command setup for Colab."""
    coordinator = ColabSetupCoordinator()
    coordinator.run_complete_setup(**kwargs)
    return coordinator


if __name__ == "__main__":
    # Test the setup when run directly
    coordinator = quick_colab_setup()
    print("\n🎉 Setup utilities ready!")
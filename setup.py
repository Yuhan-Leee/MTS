#!/usr/bin/env python3
"""
Setup script for MTS project
This script helps you set up the environment and validate configurations
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def check_gpu():
    """Check if GPU is available"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU detected: {gpu_name} ({gpu_count} available)")
            return True
        else:
            print("⚠️  No GPU detected. Training will be very slow.")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    # Core dependencies
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements-core.txt"])
        print("✅ Core dependencies installed")
    except subprocess.CalledProcessError:
        print("❌ Failed to install core dependencies")
        return False
    
    return True

def validate_data_paths():
    """Validate that data paths exist"""
    config_path = "config/default.yaml"
    
    if not os.path.exists(config_path):
        print("❌ Configuration file not found")
        return False
    
    # Read configuration
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Error reading configuration: {e}")
        return False
    
    # Check data directory
    data_dir = config['data']['input_dir']
    if os.path.exists(data_dir):
        print(f"✅ Data directory found: {data_dir}")
    else:
        print(f"❌ Data directory not found: {data_dir}")
        print("   Please update the path in config/default.yaml")
        return False
    
    # Check model directory
    model_dir = config['model']['path']
    if os.path.exists(model_dir):
        print(f"✅ Model directory found: {model_dir}")
    else:
        print(f"❌ Model directory not found: {model_dir}")
        print("   Please update the path in config/default.yaml")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "data/processed",
        "data/features", 
        "outputs/models",
        "outputs/results",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def run_quick_test():
    """Run a quick test to verify setup"""
    print("🧪 Running quick test...")
    
    try:
        # Test imports
        from data.processing import DataPreprocessor
        from data.features import FeatureExtractor
        from src.models import ThermometerModel
        from src.training import ModelEvaluator
        print("✅ All imports successful")
        
        # Test model creation
        model = ThermometerModel(4096)  # Example input dimension
        print("✅ Model creation successful")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 MTS Setup Script")
    print("=" * 50)
    
    # Change to MTS directory
    os.chdir("/data/home/zhanght/MTS")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check GPU
    check_gpu()
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Validate data paths
    if not validate_data_paths():
        print("\n⚠️  Data paths need to be configured")
        print("   Please edit config/default.yaml with your local paths")
        return False
    
    # Create directories
    create_directories()
    
    # Run quick test
    if not run_quick_test():
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Review and update config/default.yaml if needed")
    print("2. Run: python scripts/quick_start.py")
    print("3. Or run individual steps:")
    print("   - python scripts/preprocess_data.py --input_dir /path/to/mmlu --output_dir data/processed")
    print("   - python scripts/extract_features.py --model_path /path/to/model --data_dir data/processed --output_dir data/features")
    print("   - python scripts/train_models.py --features_dir data/features --output_dir outputs/results")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
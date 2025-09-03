#!/usr/bin/env python3
"""
Setup script for Domain Name LLM Evaluation Framework

This script sets up the project structure and installs dependencies.
"""

import os
import subprocess
import sys
from pathlib import Path

def create_directory_structure():
    """Create the project directory structure."""
    directories = [
        "config",
        "data/synthetic_dataset",
        "data/evaluation_data", 
        "data/edge_cases",
        "data/safety",
        "models/baseline",
        "models/improved",
        "models/evaluation",
        "src/data_generation",
        "src/model_training",
        "src/evaluation",
        "src/edge_case_discovery",
        "src/safety",
        "notebooks",
        "tests",
        "api",
        "reports"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def install_dependencies():
    """Install required Python dependencies."""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    
    return True

def create_env_example():
    """Create .env.example file."""
    env_example_content = """# API Keys for LLM-as-a-Judge evaluation
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: Weights & Biases for experiment tracking
WANDB_API_KEY=your_wandb_api_key_here

# Model configuration
MODEL_PATH=models/baseline/final
"""
    
    with open(".env.example", "w") as f:
        f.write(env_example_content)
    
    print("✅ Created .env.example file")

def create_init_files():
    """Create __init__.py files for Python packages."""
    init_dirs = [
        "src",
        "src/data_generation", 
        "src/model_training",
        "src/evaluation",
        "src/edge_case_discovery",
        "src/safety",
        "tests"
    ]
    
    for directory in init_dirs:
        init_file = Path(directory) / "__init__.py"
        init_file.touch(exist_ok=True)
        print(f"✅ Created {init_file}")

def main():
    """Main setup function."""
    print("🚀 Setting up Domain Name LLM Evaluation Framework...")
    print("=" * 60)
    
    # Create directory structure
    print("\n📁 Creating directory structure...")
    create_directory_structure()
    
    # Create __init__.py files
    print("\n🐍 Creating Python package structure...")
    create_init_files()
    
    # Create .env.example
    print("\n🔧 Creating configuration files...")
    create_env_example()
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not install_dependencies():
        print("❌ Setup incomplete due to dependency installation failure")
        return False
    
    print("\n" + "=" * 60)
    print("✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Copy .env.example to .env and add your API keys")
    print("2. Run: python src/data_generation/create_dataset.py")
    print("3. Run: python src/model_training/train_baseline.py")
    print("4. Run: python src/evaluation/run_evaluation.py")
    print("5. Open notebook.ipynb for comprehensive experiments")
    print("\n🎯 Happy experimenting!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

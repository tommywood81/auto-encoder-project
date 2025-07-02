#!/usr/bin/env python3
"""
Test script to verify W&B setup is working correctly.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_config_loader():
    """Test the config loader functionality."""
    try:
        from src.config_loader import ConfigLoader
        print("✅ ConfigLoader imported successfully!")
        
        loader = ConfigLoader()
        print("✅ ConfigLoader instantiated successfully!")
        
        config = loader.load_config('baseline')
        print("✅ Baseline config loaded successfully!")
        print(f"   Strategy: {config['features']['strategy']}")
        print(f"   Latent dim: {config['model']['latent_dim']}")
        print(f"   Learning rate: {config['model']['learning_rate']}")
        
        return True
    except Exception as e:
        print(f"❌ ConfigLoader test failed: {str(e)}")
        return False

def test_wandb_import():
    """Test W&B import."""
    try:
        import wandb
        print("✅ W&B imported successfully!")
        print(f"   Version: {wandb.__version__}")
        return True
    except Exception as e:
        print(f"❌ W&B import failed: {str(e)}")
        return False

def test_yaml_import():
    """Test YAML import."""
    try:
        import yaml
        print("✅ YAML imported successfully!")
        return True
    except Exception as e:
        print(f"❌ YAML import failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing W&B Setup")
    print("=" * 40)
    
    tests = [
        ("YAML Import", test_yaml_import),
        ("W&B Import", test_wandb_import),
        ("Config Loader", test_config_loader),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔧 Testing: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"   ❌ {test_name} failed!")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! W&B setup is ready.")
        print("\n🚀 Next steps:")
        print("   1. Run: wandb login")
        print("   2. Run: python sweep_features_wandb.py --entity your-username")
        print("   3. Or run: python main_controller.py --entity your-username")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
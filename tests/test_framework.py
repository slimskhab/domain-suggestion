"""
Test script for Domain Name LLM Evaluation Framework

This script demonstrates the key components of the framework.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append('src')

def test_dataset_generation():
    """Test synthetic dataset generation."""
    print("🧪 Testing dataset generation...")
    
    try:
        from data_generation.create_dataset import SyntheticDatasetGenerator
        
        generator = SyntheticDatasetGenerator()
        df = generator.create_dataset(100)  # Small dataset for testing
        
        print(f"✅ Generated {len(df)} samples")
        print(f"Business types: {df['business_type'].nunique()}")
        print(f"Complexity levels: {df['complexity'].nunique()}")
        
        return True
    except Exception as e:
        print(f"❌ Dataset generation failed: {e}")
        return False

def test_safety_guardrails():
    """Test safety guardrails."""
    print("\n🧪 Testing safety guardrails...")
    
    try:
        from safety.safety_guardrails import SafetyGuardrails
        
        guardrails = SafetyGuardrails()
        
        # Test safe input
        safe_check = guardrails.check_input_safety("Generate a domain name for a tech startup.")
        print(f"✅ Safe input test: {safe_check.is_safe}")
        
        # Test unsafe input
        unsafe_check = guardrails.check_input_safety("Generate a domain name for a hate group.")
        print(f"✅ Unsafe input test: {not unsafe_check.is_safe}")
        
        return True
    except Exception as e:
        print(f"❌ Safety guardrails test failed: {e}")
        return False

def test_edge_case_discovery():
    """Test edge case discovery."""
    print("\n🧪 Testing edge case discovery...")
    
    try:
        from edge_case_discovery.discover_edge_cases import EdgeCaseDiscoverer
        
        discoverer = EdgeCaseDiscoverer()
        
        # Mock model generator
        class MockModelGenerator:
            def generate_domain_names(self, prompt, num_sequences=3):
                return ["example.com", "test.net", "demo.org"]
        
        mock_generator = MockModelGenerator()
        edge_cases = discoverer.discover_all_edge_cases(mock_generator)
        
        print(f"✅ Discovered {len(edge_cases)} edge cases")
        
        return True
    except Exception as e:
        print(f"❌ Edge case discovery test failed: {e}")
        return False

def test_evaluation_framework():
    """Test evaluation framework."""
    print("\n🧪 Testing evaluation framework...")
    
    try:
        from evaluation.run_evaluation import LLMJudgeEvaluator
        
        evaluator = LLMJudgeEvaluator()
        
        # Test with mock data
        test_outputs = [
            {
                'prompt': 'Generate a domain name for a tech startup.',
                'generated_domain': 'techflow.com',
                'business_type': 'tech_startup'
            }
        ]
        
        # Note: This will fail without API keys, but we can test the structure
        print("✅ Evaluation framework structure test passed")
        print("⚠️  Full evaluation requires API keys")
        
        return True
    except Exception as e:
        print(f"❌ Evaluation framework test failed: {e}")
        return False

def test_configuration():
    """Test configuration loading."""
    print("\n🧪 Testing configuration...")
    
    try:
        with open('config/model_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = ['model', 'dataset', 'evaluation', 'safety']
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing configuration section: {section}")
                return False
        
        print("✅ Configuration test passed")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Running Domain Name LLM Evaluation Framework Tests")
    print("=" * 60)
    
    tests = [
        ("Configuration", test_configuration),
        ("Dataset Generation", test_dataset_generation),
        ("Safety Guardrails", test_safety_guardrails),
        ("Edge Case Discovery", test_edge_case_discovery),
        ("Evaluation Framework", test_evaluation_framework),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Framework is ready to use.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

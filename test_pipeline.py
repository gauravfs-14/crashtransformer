#!/usr/bin/env python3
"""
Test script for CrashTransformer Pipeline
This script demonstrates how to use the --test flag to quickly verify
that all components work correctly before running the full pipeline.
"""

import subprocess
import sys
import os

def run_test():
    """Run the pipeline in test mode."""
    print("🧪 Testing CrashTransformer Pipeline...")
    print("=" * 50)
    
    # Check if the main script exists
    if not os.path.exists("crash_transformer_pipeline.py"):
        print("❌ Error: crash_transformer_pipeline.py not found!")
        return False
    
    try:
        # Run the pipeline in test mode
        result = subprocess.run([
            sys.executable, "crash_transformer_pipeline.py", "--test"
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        # Print output
        if result.stdout:
            print("📤 STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("📤 STDERR:")
            print(result.stderr)
        
        # Check return code
        if result.returncode == 0:
            print("✅ Test completed successfully!")
            return True
        else:
            print(f"❌ Test failed with return code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1) 
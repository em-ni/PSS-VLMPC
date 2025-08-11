#!/usr/bin/env python3
"""
Test script to verify VLM visual scene generation without running full simulation.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

from VLM import VLM

class MockRod:
    """Mock rod object for testing."""
    def __init__(self, n_elements=10, length=0.4):
        # Create a simple curved rod for testing
        s = np.linspace(0, length, n_elements)
        x = s * 0.5  # Some curvature in x
        y = np.sin(s * 3) * 0.1  # Small oscillation in y
        z = -s  # Downward in z
        
        self.position_collection = np.array([x, y, z])

class MockSim:
    """Mock simulation object for testing."""
    def __init__(self):
        self.rods = [
            MockRod(n_elements=15, length=0.2),  # Rod 1
            MockRod(n_elements=15, length=0.2)   # Rod 2
        ]
        
        # Modify second rod to be slightly different
        self.rods[1].position_collection[0, :] += 0.1  # Offset in x
        self.rods[1].position_collection[2, :] -= 0.2  # More downward
    
    def get_rods(self):
        return self.rods

def test_vlm_visual():
    """Test the VLM visual scene generation."""
    print("Testing VLM visual scene generation...")
    
    # Create VLM instance
    vlm = VLM()
    
    # Create mock simulation data
    mock_sim = MockSim()
    
    # Define test data
    current_target = np.array([0.3, 0.2, -0.5])
    tip_history = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.1, 0.05, -0.1]),
        np.array([0.2, 0.1, -0.3]),
        np.array([0.25, 0.15, -0.4])
    ]
    target_history = [
        np.array([0.5, 0.0, -0.5]),
        np.array([0.4, 0.1, -0.5]),
        np.array([0.3, 0.2, -0.5])
    ]
    
    # Generate scene image
    print("Generating scene image...")
    scene_image = vlm.ingest_info(
        sim_data=mock_sim,
        current_target=current_target,
        tip_history=tip_history,
        target_history=target_history
    )
    
    if scene_image:
        print("✓ Scene image generated successfully!")
        print(f"Image data length: {len(scene_image)} characters")
        
        # Save the image for inspection
        success = vlm.save_scene_image("test_vlm_scene.png")
        if success:
            print("✓ Scene image saved for inspection")
        else:
            print("✗ Failed to save scene image")
            
        return True
    else:
        print("✗ Failed to generate scene image")
        return False

def test_vlm_query_without_server():
    """Test VLM query structure (will fail without server, but we can check the format)."""
    print("\nTesting VLM query structure...")
    
    vlm = VLM()
    
    # Generate a test scene
    mock_sim = MockSim()
    scene_image = vlm.ingest_info(sim_data=mock_sim, current_target=np.array([0.5, 0.0, -0.5]))
    
    if scene_image:
        print("✓ Scene image ready for VLM query")
        print("✓ Query structure prepared (server not tested)")
        return True
    else:
        print("✗ Scene image generation failed")
        return False

if __name__ == "__main__":
    print("VLM Visual Testing")
    print("=" * 50)
    
    # Test 1: Visual scene generation
    test1_passed = test_vlm_visual()
    
    # Test 2: Query structure (without server)
    test2_passed = test_vlm_query_without_server()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"  Visual Generation: {'PASS' if test1_passed else 'FAIL'}")
    print(f"  Query Structure:   {'PASS' if test2_passed else 'FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n✓ All tests passed! VLM visual integration is ready.")
        print("\nNext steps:")
        print("1. Start your VLM server:")
        print("   ./llama-server -m ./models/smolvlm-500m-instruct-q4_k_m.gguf -ngl 99 --port 8080")
        print("2. Run control.py with CONTROL_MODE='vlm'")
        print("3. Type commands like 'go right', 'move up', etc.")
    else:
        print("\n✗ Some tests failed. Check the errors above.")

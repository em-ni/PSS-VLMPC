#!/usr/bin/env python3
"""
Simple VLM test script to debug the text-only functionality
"""

import sys
import os

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

from VLM import VLM
import numpy as np

def test_text_only_vlm():
    """Test VLM in text-only mode"""
    print("Testing VLM in text-only mode...")
    
    # Create VLM in text-only mode
    vlm = VLM(text_only_mode=True)
    
    # Check server connection
    if not vlm.check_server():
        print("✗ VLM server not running. Start with:")
        print("./llama-server -m ./models/smolvlm-500m-instruct-q4_k_m.gguf -ngl 99 --port 8080")
        return False
    
    print("✓ VLM server connected")
    
    # Test commands
    test_commands = [
        "go right",
        "move left", 
        "up",
        "down",
        "center",
        "stop",
        "nonsense command"
    ]
    
    current_state = np.array([0.0, 0.0, -0.8, 0.0, 0.0, 0.0])
    
    for cmd in test_commands:
        print(f"\nTesting: '{cmd}'")
        
        # Simulate user input
        vlm.user_input_queue.put(cmd)
        
        # Process the input
        trajectory, target_name = vlm.process_user_input(current_state, scene_image=None)
        
        if trajectory is not None:
            print(f"✓ Response: {target_name}")
        else:
            print("✗ No valid response")
    
    print("\nText-only VLM test completed")
    return True

if __name__ == "__main__":
    test_text_only_vlm()

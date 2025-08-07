# src/VLM.py
import requests
import threading
import time
import numpy as np
from queue import Queue, Empty
import sys
import select

"""
brew install llama.cpp
llama-server -hf ggml-org/SmolVLM-500M-Instruct-GGUF -ngl 99 --port 8080
"""

class VLM:
    def __init__(self, server_url="http://localhost:8080", vlm_dt=1.0, mpc_dt=0.02):
        """
        Vision Language Model interface for dynamic target assignment.
        
        Args:
            server_url (str): URL of the llama.cpp server
            vlm_dt (float): VLM update frequency in seconds
            mpc_dt (float): MPC update frequency in seconds
        """
        self.server_url = server_url
        self.vlm_dt = vlm_dt
        self.mpc_dt = mpc_dt
        self.session = requests.Session()
        
        # Predefined targets
        self.targets = {
            'right': np.array([0.5, 0.0, -0.5, 0.0, 0.0, 0.0]),
            'left': np.array([-0.5, 0.0, -0.5, 0.0, 0.0, 0.0]),
            'up': np.array([0.0, 0.5, -0.5, 0.0, 0.0, 0.0]),
            'down': np.array([0.0, -0.5, -0.5, 0.0, 0.0, 0.0]),
            'center': np.array([0.0, 0.0, -0.8, 0.0, 0.0, 0.0])
        }
        
        # State variables
        self.current_target = None
        self.current_trajectory = None
        self.processing = False
        self.last_response = "VLM initialized. Type commands like 'go right', 'move left', etc."
        
        # Input handling
        self.user_input_queue = Queue()
        self.input_thread = None
        self.running = False
        
        # System prompt for the VLM
        self.system_prompt = """You are a robot control assistant. Users will give you movement commands for a robotic system.
You can ONLY responde with a target direction intended by the user from the following options:
- right
- left
- up
- down
- center
- stop
No puntuation, no additional text, no caps lock, no emojis, no special characters.
Respond with the exact target direction as a single word.
If the command is not recognized, respond with stop
"""

    def check_server(self):
        """Check if the llama.cpp server is running."""
        try:
            response = self.session.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def query_vlm(self, user_input):
        """Query the VLM with user input to get target direction."""
        if self.processing:
            return None
            
        self.processing = True
        try:
            payload = {
                "model": "gpt-4-vision-preview",  # This is just for compatibility
                "max_tokens": 50,
                "temperature": 1,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_input}
                ]
            }
            
            response = self.session.post(
                f"{self.server_url}/v1/chat/completions", 
                json=payload, 
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                result = data["choices"][0]["message"]["content"].strip().lower()
                self.last_response = f"VLM response: '{result}' from input: '{user_input}'"
                return result
            else:
                error_msg = f"Server error {response.status_code}"
                self.last_response = error_msg
                return None
                
        except Exception as e:
            error_msg = f"VLM Error: {str(e)}"
            self.last_response = error_msg
            return None
        finally:
            self.processing = False

    def generate_trajectory(self, current_state, target_state, transition_time=5.0):
        """
        Generate a smooth trajectory from current state to target state.
        
        Args:
            current_state (np.array): Current 6D state [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
            target_state (np.array): Target 6D state
            transition_time (float): Time to reach target in seconds
            
        Returns:
            np.array: Trajectory array of shape (n_steps, 6)
        """
        n_steps = int(transition_time / self.mpc_dt)
        if n_steps < 1:
            n_steps = 1
            
        # Generate smooth trajectory (linear interpolation for now)
        trajectory = np.zeros((n_steps, 6))
        for i in range(n_steps):
            alpha = i / (n_steps - 1) if n_steps > 1 else 1.0
            # Smooth interpolation using sigmoid-like function
            smooth_alpha = 3 * alpha**2 - 2 * alpha**3  # Smoothstep function
            trajectory[i] = current_state + smooth_alpha * (target_state - current_state)
            
        return trajectory

    def start_input_thread(self):
        """Start the input monitoring thread."""
        self.running = True
        self.input_thread = threading.Thread(target=self._input_worker, daemon=True)
        self.input_thread.start()
        print("VLM input thread started. Type commands during simulation!")

    def _input_worker(self):
        """Worker thread to handle user input."""
        while self.running:
            try:
                # Non-blocking input check
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    user_input = input().strip()
                    if user_input:
                        self.user_input_queue.put(user_input)
                        print(f"Command queued: '{user_input}'")
            except Exception as e:
                # Handle input errors gracefully
                pass

    def process_user_input(self, current_state):
        """
        Process any pending user input and return new trajectory if needed.
        
        Args:
            current_state (np.array): Current robot state
            
        Returns:
            tuple: (new_trajectory, target_name) or (None, None) if no new command
        """
        try:
            # Check for new user input
            user_input = self.user_input_queue.get_nowait()
            print(f"Processing command: '{user_input}'")
            
            # Query VLM
            vlm_response = self.query_vlm(user_input)
            
            if vlm_response is None:
                print("VLM query failed")
                return None, None
                
            # Handle stop command
            if vlm_response == "stop":
                print("Stop command received")
                # Create a trajectory that stays at current position
                stop_target = current_state.copy()
                stop_target[3:] = 0.0  # Zero velocities
                trajectory = self.generate_trajectory(current_state, stop_target, 0.5)
                return trajectory, "stop"
            
            # Handle movement commands
            if vlm_response in self.targets:
                target_state = self.targets[vlm_response]
                print(f"Moving to: {vlm_response} -> {target_state[:3]}")
                
                # Generate trajectory
                trajectory = self.generate_trajectory(current_state, target_state)
                self.current_target = vlm_response
                self.current_trajectory = trajectory
                
                return trajectory, vlm_response
            else:
                print(f"Unknown command: '{vlm_response}'")
                return None, None
                
        except Empty:
            # No new input
            return None, None
        except Exception as e:
            print(f"Error processing input: {e}")
            return None, None

    def get_status(self):
        """Get current VLM status information."""
        return {
            'processing': self.processing,
            'current_target': self.current_target,
            'last_response': self.last_response,
            'server_connected': self.check_server(),
            'queue_size': self.user_input_queue.qsize()
        }

    def stop(self):
        """Stop the VLM and cleanup."""
        self.running = False
        if self.input_thread and self.input_thread.is_alive():
            self.input_thread.join(timeout=1.0)
        print("VLM stopped")

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop()
# src/VLM.py
import requests
import threading
import numpy as np
from queue import Queue, Empty
import sys
import select
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import base64
import time

"""
brew install llama.cpp
llama-server -hf ggml-org/SmolVLM-500M-Instruct-GGUF -ngl 99 --port 8080
"""

class VLM:
    def __init__(self, server_url="http://localhost:8080", vlm_dt=1.0, mpc_dt=0.02, text_only_mode=False):
        """
        Vision Language Model interface for dynamic target assignment.
        
        Args:
            server_url (str): URL of the llama.cpp server
            vlm_dt (float): VLM update frequency in seconds
            mpc_dt (float): MPC update frequency in seconds
            text_only_mode (bool): If True, skip image generation for debugging
        """
        self.server_url = server_url
        self.vlm_dt = vlm_dt
        self.mpc_dt = mpc_dt
        self.session = requests.Session()
        self.text_only_mode = text_only_mode
        
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
        self.current_scene_image = None  # Store the latest scene image
        
        # Input handling
        self.user_input_queue = Queue()
        self.input_thread = None
        self.running = False
        
        # Visualization parameters for scene reconstruction
        self.xlim = (-1.5, 1.5)
        self.ylim = (-1.5, 1.5)
        self.zlim = (-1.5, 1.5)
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
        
        # System prompt for the VLM
        if self.text_only_mode:
            self.system_prompt = """You are a robot controller. Respond with EXACTLY ONE WORD.

Valid responses:
right
left
up
down
center
stop

CRITICAL RULES:
- NO quotes, NO periods, NO punctuation
- NO explanations or descriptions
- ONLY one word from the list above
- Do not add anything else

Examples:
"go right" → right
"move up" → up
"unclear" → stop

Just the word, nothing else."""
        else:
            self.system_prompt = """You are a robot controller. You see an image showing a robot system.

In the image:
- Red/Blue lines = Robot segments
- Red circle = Robot tip (current position)
- Green circle = Target position
- Dashed line = Distance to target

Respond with EXACTLY ONE WORD based on the user command:

Valid responses:
right
left  
up
down
center
stop

CRITICAL RULES:
- NO quotes, NO periods, NO punctuation
- NO descriptions of the image
- ONLY respond with one word from the list above

Examples:
"go right" → right
"move to target" → (look at image and decide direction)
"unclear" → stop"""

    def check_server(self):
        """Check if the llama.cpp server is running."""
        try:
            response = self.session.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def ingest_info(self, sim_data, current_target=None, tip_history=None, target_history=None):
        """
        Create visual representation of the current scene for VLM processing.
        
        Args:
            sim_data: Simulation object or rods data
            current_target (np.array): Current target position [x, y, z]
            tip_history (list): History of tip positions
            target_history (list): History of target positions
            
        Returns:
            str: Base64 encoded image of the scene
        """
        try:
            # Get rods data from simulation
            if hasattr(sim_data, 'get_rods'):
                rods = sim_data.get_rods()
            else:
                rods = sim_data
                
            if not rods:
                return None
                
            # Create a simpler, single view figure for better VLM understanding
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            
            ax.set_xlim(self.xlim)
            ax.set_ylim(self.zlim)  # Use Z axis for vertical movement
            ax.set_xlabel('X Position (m)', fontsize=14)
            ax.set_ylabel('Z Position (m)', fontsize=14)  
            ax.set_title('Robot Control View - XZ Plane', fontsize=16, fontweight='bold')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # Plot robot rods in XZ plane (side view)
            for i, rod in enumerate(rods):
                color = self.colors[i % len(self.colors)]
                pos = rod.position_collection
                
                # XZ view (side view)
                ax.plot(pos[0, :], pos[2, :], color=color, linewidth=4, 
                       marker='o', markersize=4, label=f'Robot Segment {i+1}')
            
            # Get current tip position
            current_tip = None
            if len(rods) > 0:
                current_tip = rods[-1].position_collection[:, -1]
                
                # Highlight tip with larger marker
                ax.plot(current_tip[0], current_tip[2], 'ro', markersize=12, 
                       markeredgecolor='darkred', markeredgewidth=2, 
                       label='ROBOT TIP', zorder=10)
            
            # Plot current target if provided
            if current_target is not None:
                ax.plot(current_target[0], current_target[2], 'go', markersize=15, 
                       markeredgecolor='darkgreen', markeredgewidth=3, 
                       label='TARGET', zorder=10)
                
                # Draw connection line between tip and target
                if current_tip is not None:
                    ax.plot([current_tip[0], current_target[0]], [current_tip[2], current_target[2]], 
                           'k--', linewidth=3, alpha=0.8, label='Distance')
                    
                    # Calculate and display distance prominently
                    distance = np.linalg.norm(current_tip - current_target)
                    ax.text(0.05, 0.95, f'Distance to Target: {distance:.3f}m', 
                           transform=ax.transAxes, fontsize=12, fontweight='bold',
                           verticalalignment='top', 
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
                    
                    # Show direction arrow
                    dx = current_target[0] - current_tip[0]
                    dz = current_target[2] - current_tip[2]
                    if abs(dx) > 0.01 or abs(dz) > 0.01:  # Only show if there's meaningful distance
                        ax.annotate('', xy=(current_target[0], current_target[2]), 
                                   xytext=(current_tip[0], current_tip[2]),
                                   arrowprops=dict(arrowstyle='->', lw=3, color='purple', alpha=0.8))
            
            # Plot tip trajectory history (simplified)
            if tip_history and len(tip_history) > 1:
                tip_hist = np.array(tip_history)
                ax.plot(tip_hist[:, 0], tip_hist[:, 2], 'b-', linewidth=2, alpha=0.6, 
                       label='Tip Path')
            
            # Add movement direction indicators
            ax.text(0.8, 0.8, 'RIGHT →', transform=ax.transAxes, fontsize=10, ha='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            ax.text(0.2, 0.8, '← LEFT', transform=ax.transAxes, fontsize=10, ha='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            ax.text(0.5, 0.9, '↑ UP', transform=ax.transAxes, fontsize=10, ha='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            ax.text(0.5, 0.1, '↓ DOWN', transform=ax.transAxes, fontsize=10, ha='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            
            # Add legend
            ax.legend(loc='upper left', fontsize=10)
            
            plt.tight_layout()
            
            # Convert to base64 encoded image
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')  # Lower DPI for faster processing
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            plt.close(fig)  # Clean up
            
            # Store for later use
            self.current_scene_image = image_base64
            
            return image_base64
            
        except Exception as e:
            print(f"Error creating scene image: {e}")
            return None

    def query_vlm(self, user_input, scene_image=None):
        """Query the VLM with user input and optional scene image to get target direction."""
        if self.processing:
            return None
            
        self.processing = True
        try:
            # Prepare the message content
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]
            
            # Create user message with text and optional image
            user_message = {"role": "user", "content": []}
            
            # Add text content
            user_message["content"].append({
                "type": "text",
                "text": user_input
            })
            
            # Add image if provided
            if scene_image or self.current_scene_image:
                image_data = scene_image or self.current_scene_image
                user_message["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_data}"
                    }
                })
            
            messages.append(user_message)
            
            payload = {
                "model": "gpt-4-vision-preview",  # This is just for compatibility
                "max_tokens": 10,  # Reduced to force shorter responses
                "temperature": 0.1,  # Lower temperature for more consistent responses
                "messages": messages
            }
            
            print(f"Sending VLM query: '{user_input}' with {'image' if (scene_image or self.current_scene_image) else 'text only'}")
            
            response = self.session.post(
                f"{self.server_url}/v1/chat/completions", 
                json=payload, 
                timeout=15  # Slightly longer timeout for image processing
            )
            
            print(f"VLM server response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                raw_result = data["choices"][0]["message"]["content"].strip()
                
                print(f"VLM raw response: '{raw_result}'")
                
                # Clean up the response - remove quotes, periods, and extra characters
                cleaned_result = raw_result.lower()
                cleaned_result = cleaned_result.replace('"', '').replace("'", "")
                cleaned_result = cleaned_result.replace('.', '').replace(',', '').replace('!', '').replace('?', '')
                cleaned_result = cleaned_result.strip()
                
                # Extract only the first word if response is too long
                first_word = cleaned_result.split()[0] if cleaned_result.split() else cleaned_result
                
                print(f"VLM cleaned response: '{first_word}'")
                
                image_info = " (with image)" if (scene_image or self.current_scene_image) else " (text only)"
                self.last_response = f"VLM response: '{first_word}' from input: '{user_input}'{image_info}"
                return first_word
            else:
                error_msg = f"Server error {response.status_code}: {response.text}"
                print(f"VLM server error: {error_msg}")
                self.last_response = error_msg
                return None
                
        except Exception as e:
            error_msg = f"VLM Error: {str(e)}"
            print(f"VLM exception: {error_msg}")
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

    def process_user_input(self, current_state, scene_image=None):
        """
        Process any pending user input and return new trajectory if needed.
        
        Args:
            current_state (np.array): Current robot state
            scene_image (str): Base64 encoded scene image (optional)
            
        Returns:
            tuple: (new_trajectory, target_name) or (None, None) if no new command
        """
        try:
            # Check for new user input
            user_input = self.user_input_queue.get_nowait()
            print(f"Processing command: '{user_input}'")
            
            # First try with scene image if available
            vlm_response = None
            if scene_image:
                vlm_response = self.query_vlm(user_input, scene_image)
                
            # If image query failed or no image, try text-only
            if vlm_response is None:
                print("Trying text-only VLM query...")
                vlm_response = self.query_vlm(user_input, scene_image=None)
            
            if vlm_response is None:
                print("VLM query failed completely")
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
                print(f"Unknown command: '{vlm_response}', trying to extract valid direction...")
                
                # Try to extract a valid direction from the response
                for direction in self.targets.keys():
                    if direction in vlm_response.lower():
                        print(f"Extracted direction: {direction}")
                        target_state = self.targets[direction]
                        trajectory = self.generate_trajectory(current_state, target_state)
                        self.current_target = direction
                        self.current_trajectory = trajectory
                        return trajectory, direction
                
                print("No valid direction found, defaulting to stop")
                return None, None
                
        except Empty:
            # No new input
            return None, None
        except Exception as e:
            print(f"Error processing input: {e}")
            return None, None

    def save_scene_image(self, filename=None):
        """Save the current scene image to disk for debugging."""
        if self.current_scene_image is None:
            print("No scene image available to save")
            return False
            
        try:
            import os
            if filename is None:
                filename = f"vlm_scene_{int(time.time())}.png"
                
            # Ensure results directory exists
            os.makedirs("results", exist_ok=True)
            filepath = os.path.join("results", filename)
            
            # Decode and save image
            image_data = base64.b64decode(self.current_scene_image)
            with open(filepath, 'wb') as f:
                f.write(image_data)
                
            print(f"Scene image saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"Error saving scene image: {e}")
            return False

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
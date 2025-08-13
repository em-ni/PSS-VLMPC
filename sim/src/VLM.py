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
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

"""
For Llama.cpp:
brew install llama.cpp
llama-server -hf ggml-org/SmolVLM-500M-Instruct-GGUF -ngl 99 --port 8080

For Gemini:
pip install google-genai python-dotenv
Add G_API_KEY=your-api-key to .env file
"""

class VLM:
    def __init__(self, server_url="http://localhost:8080", vlm_dt=1.0, mpc_dt=0.02, backend="llama", model_name="gemini-2.5-pro"):
        """
        Vision Language Model interface for dynamic target assignment.
        
        Args:
            server_url (str): URL of the llama.cpp server (for backend="llama")
            vlm_dt (float): VLM update frequency in seconds
            mpc_dt (float): MPC update frequency in seconds
            backend (str): "llama" for llama.cpp server or "gemini" for Google Gemini
            model_name (str): Model name (for Gemini: "gemini-2.5-pro" or "gemini-2.5-flash")
        """
        self.server_url = server_url
        self.vlm_dt = vlm_dt
        self.mpc_dt = mpc_dt
        self.backend = backend.lower()
        self.model_name = model_name
        
        # Initialize default attributes first (before any potential exceptions)
        self.session = None
        self.gemini_client = None
        self.user_input_queue = Queue()
        self.input_thread = None
        self.running = False
        
        # Initialize backend-specific clients
        if self.backend == "llama":
            self.session = requests.Session()
        elif self.backend == "gemini":
            # Look for .env file in current directory and parent directories
            env_paths = [
                '.env',
                '../.env', 
                '../../.env',
                os.path.join(os.path.dirname(__file__), '..', '.env'),
                os.path.join(os.path.dirname(__file__), '..', '..', '.env')
            ]
            
            for env_path in env_paths:
                if os.path.exists(env_path):
                    load_dotenv(env_path)
                    break
            
            os.environ['GOOGLE_API_KEY'] = os.getenv('G_API_KEY')
            self.gemini_client = genai.Client()
        else:
            raise ValueError(f"Unsupported backend: {backend}. Use 'llama' or 'gemini'")
        
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
        
        # Visualization parameters for scene reconstruction
        self.xlim = (-1.0, 1.0)
        self.ylim = (-1.0, 1.0)
        self.zlim = (-1.0, 1.0)
        self.colors = ['red', 'blue', 'orange', 'purple']
        
        # System prompt for the VLM
        self.system_prompt = """You are a soft robot controller, your goal is to assign the 2D coordinates of the target, where the tip of the robot will be steered after your output. You see the XY plane of the robot's workspace, you can see the robot tip colored in as a black dot, and the current target colored as a green dot. You are able to assign the position of the latter, and the tip possiotn will be controlled to follow the target. Respond with EXACTLY with two numbers separated by a comma.
You have to understand the user intention, and output the target position in the XY plane WITHIN the workspace limits. For example the use might want the robot to touch a certain object in the workspace, or to move to a certain position, or to move in a certain direction. 
Possible examples:
"touch the red circle" → 0.5,0.5 (coordinates of the red circle you deduced from the image)
"move right" → 0.5,0.0 (you deduce the tip position is at (0,0) and you set the target to (0.5,0))
"avoid the yellow square" → 0.0,-0.5 (you deduce the yellow square is at (0, -0.5), you see the tip is close to it and you set the target away from it (0, 0.5))

CRITICAL RULES:
- NO quotes, NO periods, NO punctuation
- NO explanations or descriptions
- ONLY the two numbers separated by a comma
- The numbers must be within the workspace limits: X in [-1.0, 1.0], Y in [-1.0, 1.0]
- The first number is the X coordinate, the second number is the Y coordinate
- Do not add anything else
"""

    def check_server(self):
        """Check if the backend is available."""
        if self.backend == "llama":
            try:
                response = self.session.get(f"{self.server_url}/health", timeout=5)
                return response.status_code == 200
            except:
                return False
        elif self.backend == "gemini":
            try:
                # Only try connection test if we have a valid client
                if self.gemini_client is None:
                    return False
                    
                # Simple test to check if Gemini is accessible
                test_response = self.gemini_client.models.generate_content(
                    model=self.model_name,
                    contents=["Test connection"],
                )
                return True
            except Exception as e:
                print(f"Gemini connection test failed: {e}")
                return False
        return False

    def ingest_info(self, sim_data, current_target=None, tip_history=None, target_history=None):
        """
        Create visual representation of the current scene for VLM processing.
        
        Args:
            sim_data: Simulation object
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
            ax.set_ylim(self.ylim)
            ax.set_aspect('equal', adjustable='box')
            ax.set_title("Robot Workspace Scene XY View")
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")
            ax.grid(True)
            
            # Plot robot tip position in black
            current_tip = rods[-1].position_collection[:,-1]
            ax.plot(current_tip[0], current_tip[1], 'ko', markersize=10, 
                   markeredgecolor='black', markeredgewidth=2, label='ROBOT TIP', zorder=5)
            
            # Plot current target position in green
            if current_target is not None:
                ax.plot(current_target[0], current_target[1], 'go', markersize=10, 
                       markeredgecolor='black', markeredgewidth=2, label='CURRENT TARGET', zorder=5)
            else:
                current_target = np.array([0.0, 0.0])
            
            # Extract and plot targets directly from simulation
            if hasattr(sim_data, 'get_targets'):
                sim_targets = sim_data.get_targets()
                for target in sim_targets:
                    # Get target position
                    if hasattr(target, 'position_collection'):
                        if target.position_collection.ndim == 1:
                            target_pos = target.position_collection
                        else:
                            target_pos = target.position_collection[:, 0]
                    else:
                        continue  # Skip if no position data
                    
                    # Get target color (use the stored target_color attribute)
                    target_color = getattr(target, 'target_color', 'gray')
                    target_id = getattr(target, 'target_id', 'unknown')
                    
                    # Plot the target with its correct color
                    ax.plot(target_pos[0], target_pos[1], 'o', color=target_color,
                           markersize=8, markeredgecolor='black', markeredgewidth=1,
                           label=f'Target {target_id}: {target_color.upper()}', zorder=4)

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
            if self.backend == "llama":
                return self._query_llama(user_input, scene_image)
            elif self.backend == "gemini":
                return self._query_gemini(user_input, scene_image)
        except Exception as e:
            error_msg = f"VLM Error: {str(e)}"
            print(f"VLM exception: {error_msg}")
            self.last_response = error_msg
            return None
        finally:
            self.processing = False

    def _query_llama(self, user_input, scene_image=None):
        """Query Llama.cpp server."""
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
        
        print(f"Sending Llama VLM query: '{user_input}' with {'image' if (scene_image or self.current_scene_image) else 'text only'}")
        
        response = self.session.post(
            f"{self.server_url}/v1/chat/completions", 
            json=payload, 
            timeout=15  # Slightly longer timeout for image processing
        )
        
        print(f"Llama server response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            raw_result = data["choices"][0]["message"]["content"].strip()
            
            print(f"Llama raw response: '{raw_result}'")
            
            # Clean up the response - remove quotes, periods, and extra characters
            cleaned_result = raw_result.lower()
            cleaned_result = cleaned_result.replace('"', '').replace("'", "")
            cleaned_result = cleaned_result.replace('.', '').replace(',', '').replace('!', '').replace('?', '')
            cleaned_result = cleaned_result.strip()
            
            # Extract only the first word if response is too long
            first_word = cleaned_result.split()[0] if cleaned_result.split() else cleaned_result
            
            print(f"Llama cleaned response: '{first_word}'")
            
            image_info = " (with image)" if (scene_image or self.current_scene_image) else " (text only)"
            self.last_response = f"Llama response: '{first_word}' from input: '{user_input}'{image_info}"
            return first_word
        else:
            error_msg = f"Llama server error {response.status_code}: {response.text}"
            print(f"Llama server error: {error_msg}")
            self.last_response = error_msg
            return None

    def _query_gemini(self, user_input, scene_image=None):
        """Query Google Gemini. Return target coordinates as a string: x,y"""
        print(f"Sending Gemini query: '{user_input}' with {'image' if (scene_image or self.current_scene_image) else 'text only'}")
        
        # Prepare contents for Gemini
        contents = []
        
        # Add system instruction through the user prompt for now
        # (Gemini has system instructions but this approach is simpler)
        full_prompt = f"{self.system_prompt}\n\nUser command: {user_input}"
        contents.append(full_prompt)
        
        # Add image if provided and not in text-only mode
        if (scene_image or self.current_scene_image):
            image_data = scene_image or self.current_scene_image
            
            # Convert base64 to bytes for Gemini
            try:
                image_bytes = base64.b64decode(image_data)
                image_part = types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/png'
                )
                contents.append(image_part)
            except Exception as e:
                print(f"Error processing image for Gemini: {e}")
                # Continue without image
        
        try:
            response = self.gemini_client.models.generate_content(
                model=self.model_name,
                contents=contents,
            )
            self.last_response = response.text
            raw_result = response.text
            print(f"Gemini raw response: {raw_result}")
            
            # Check if it is in the expected format: x,y
            # Check if response is in x,y format
            if ',' in raw_result:
                try:
                    # Try to parse as x,y coordinates
                    parts = raw_result.strip().split(',')
                    if len(parts) == 2:
                        x = float(parts[0].strip())
                        y = float(parts[1].strip())
                        
                        # Validate coordinates are within workspace limits
                        if self.xlim[0] <= x <= self.xlim[1] and self.ylim[0] <= y <= self.ylim[1]:
                            print(f"Gemini coordinates parsed: ({x}, {y})")
                            target = f"{x},{y}"
                        else:
                            print(f"Coordinates out of bounds: ({x}, {y}), defaulting to center")
                            target = "0.0,0.0"
                    else:
                        print(f"Invalid coordinate format, defaulting to center")
                        target = "0.0,0.0"
                except ValueError:
                    print(f"Could not parse coordinates, defaulting to center")
                    target = "0.0,0.0"
            else:
                # Not right format default to center target
                print(f"Response not in x,y format: '{raw_result}', defaulting to center")
                target = "0.0,0.0"
                
            return target
            
        except Exception as e:
            error_msg = f"Gemini API error: {str(e)}"
            print(f"Gemini error: {error_msg}")
            self.last_response = error_msg
            return "0.0,0.0"

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
            try:
                coords = vlm_response.split(',')
                if len(coords) == 2:
                    x = float(coords[0].strip())
                    y = float(coords[1].strip())
                    
                    # Validate coordinates
                    if self.xlim[0] <= x <= self.xlim[1] and self.ylim[0] <= y <= self.ylim[1]:
                        target_state = np.array([x, y, -0.5, 0.0, 0.0, 0.0])  # Z and velocities are zero
                        print(f"Moving to coordinates: ({x}, {y})")
                        
                        # Generate trajectory
                        trajectory = self.generate_trajectory(current_state, target_state)
                        self.current_target = f"{x},{y}"
                        self.current_trajectory = trajectory
                        
                        return trajectory, f"{x},{y}"
                    else:
                        print(f"Coordinates out of bounds: ({x}, {y}), defaulting to center")
                        return None, None
                else:
                    print(f"Invalid coordinate format: '{vlm_response}', defaulting to center")
                    return None, None
            except ValueError:
                print(f"Could not parse coordinates from response: '{vlm_response}', defaulting to center")
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
            'backend': self.backend,
            'model_name': self.model_name if self.backend == "gemini" else "llama.cpp",
            'server_connected': self.check_server(),
            'queue_size': self.user_input_queue.qsize()
        }

    def stop(self):
        """Stop the VLM and cleanup."""
        self.running = False
        if hasattr(self, 'input_thread') and self.input_thread and self.input_thread.is_alive():
            self.input_thread.join(timeout=1.0)
        print("VLM stopped")

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.stop()
        except Exception:
            # Ignore errors during cleanup
            pass


    
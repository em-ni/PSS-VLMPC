import time
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

class RTPlotter:
    def __init__(self, rods, xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), zlim=(-1.5, 1.5), 
                 show_reference=True, trajectory_history_length=50):
        self.rods = rods
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.show_reference = show_reference
        self.trajectory_history_length = trajectory_history_length
        
        # Storage for reference trajectory and history
        self.reference_position = None
        self.reference_history = deque(maxlen=trajectory_history_length)
        self.tip_history = deque(maxlen=trajectory_history_length)
        self.connecting_line_history = deque(maxlen=trajectory_history_length)  # Store connecting line points
        
        # Track if we need to draw the connecting line from start to trajectory
        self.trajectory_started = False
        self.start_tip_position = None
        
        # Setup the figure and axes
        plt.ion()  # Turn on interactive mode
        self.fig = plt.figure(figsize=(18, 6))
        
        # Create subplots for different views
        self.ax_xy = self.fig.add_subplot(131)
        self.ax_xz = self.fig.add_subplot(132)
        self.ax_3d = self.fig.add_subplot(133, projection='3d')
        
        # Set limits and labels
        self.ax_xy.set_xlim(xlim)
        self.ax_xy.set_ylim(ylim)
        self.ax_xy.set_xlabel('X (m)')
        self.ax_xy.set_ylabel('Y (m)')
        self.ax_xy.set_title('XY View')
        self.ax_xy.set_aspect('equal')
        self.ax_xy.grid(True, alpha=0.3)
        
        self.ax_xz.set_xlim(xlim)
        self.ax_xz.set_ylim(zlim)
        self.ax_xz.set_xlabel('X (m)')
        self.ax_xz.set_ylabel('Z (m)')
        self.ax_xz.set_title('XZ View')
        self.ax_xz.set_aspect('equal')
        self.ax_xz.grid(True, alpha=0.3)
        
        self.ax_3d.set_xlim(xlim)
        self.ax_3d.set_ylim(ylim)
        self.ax_3d.set_zlim(zlim)
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        self.ax_3d.set_title('3D View')
        
        # Initialize line objects for each rod
        self.lines_xy = []
        self.lines_xz = []
        self.lines_3d = []
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
        
        for i, rod in enumerate(self.rods):
            color = colors[i % len(colors)]
            
            # XY view lines
            line_xy, = self.ax_xy.plot([], [], color=color, linewidth=3, 
                                     marker='o', markersize=3, label=f'Rod {i+1}')
            self.lines_xy.append(line_xy)
            
            # XZ view lines
            line_xz, = self.ax_xz.plot([], [], color=color, linewidth=3, 
                                     marker='o', markersize=3, label=f'Rod {i+1}')
            self.lines_xz.append(line_xz)
            
            # 3D view lines
            line_3d, = self.ax_3d.plot([], [], [], color=color, linewidth=3, 
                                     marker='o', markersize=4, label=f'Rod {i+1}')
            self.lines_3d.append(line_3d)
        
        # Initialize reference trajectory visualization elements
        if self.show_reference:
            # Current reference target (point slightly bigger than trajectory line)
            self.ref_target_xy, = self.ax_xy.plot([], [], 'go', markersize=8, 
                                                 markeredgecolor='darkgreen', markeredgewidth=1,
                                                 label='Target', zorder=10)
            self.ref_target_xz, = self.ax_xz.plot([], [], 'go', markersize=8, 
                                                 markeredgecolor='darkgreen', markeredgewidth=1,
                                                 label='Target', zorder=10)
            self.ref_target_3d, = self.ax_3d.plot([], [], [], 'go', markersize=8, 
                                                 markeredgecolor='darkgreen', markeredgewidth=1,
                                                 label='Target', zorder=10)
            
            # Reference trajectory history (green solid line)
            self.ref_history_xy, = self.ax_xy.plot([], [], 'g-', linewidth=3, alpha=0.8,
                                                   label='Target Path')
            self.ref_history_xz, = self.ax_xz.plot([], [], 'g-', linewidth=3, alpha=0.8,
                                                   label='Target Path')
            self.ref_history_3d, = self.ax_3d.plot([], [], [], 'g-', linewidth=3, alpha=0.8,
                                                   label='Target Path')
            
            # Connecting line from start tip to trajectory start (dotted green)
            self.connecting_line_xy, = self.ax_xy.plot([], [], 'g:', linewidth=2, alpha=0.7,
                                                      label='Connection')
            self.connecting_line_xz, = self.ax_xz.plot([], [], 'g:', linewidth=2, alpha=0.7,
                                                      label='Connection')
            self.connecting_line_3d, = self.ax_3d.plot([], [], [], 'g:', linewidth=2, alpha=0.7,
                                                      label='Connection')
            
            # Tip trajectory history (blue solid line)
            self.tip_history_xy, = self.ax_xy.plot([], [], 'b-', linewidth=2, alpha=0.8,
                                                   label='Tip Path')
            self.tip_history_xz, = self.ax_xz.plot([], [], 'b-', linewidth=2, alpha=0.8,
                                                   label='Tip Path')
            self.tip_history_3d, = self.ax_3d.plot([], [], [], 'b-', linewidth=2, alpha=0.8,
                                                   label='Tip Path')
            
            # Current connection line between target and tip (thin black dotted)
            self.current_connection_xy, = self.ax_xy.plot([], [], 'k:', linewidth=1, alpha=0.5)
            self.current_connection_xz, = self.ax_xz.plot([], [], 'k:', linewidth=1, alpha=0.5)
            self.current_connection_3d, = self.ax_3d.plot([], [], [], 'k:', linewidth=1, alpha=0.5)
        
        # Add legends
        self.ax_xy.legend(loc='upper right', bbox_to_anchor=(1, 1))
        self.ax_xz.legend(loc='upper right', bbox_to_anchor=(1, 1))
        self.ax_3d.legend(loc='upper right', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        plt.show(block=False)
        
        # Add text for tracking error display
        if self.show_reference:
            self.error_text_xy = self.ax_xy.text(0.02, 0.98, '', transform=self.ax_xy.transAxes,
                                               verticalalignment='top', fontsize=10,
                                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            self.error_text_xz = self.ax_xz.text(0.02, 0.98, '', transform=self.ax_xz.transAxes,
                                               verticalalignment='top', fontsize=10,
                                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            self.error_text_3d = self.ax_3d.text2D(0.02, 0.98, '', transform=self.ax_3d.transAxes,
                                                  verticalalignment='top', fontsize=10,
                                                  bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
    def set_reference_target(self, target_position):
        """Set the current reference target position"""
        self.reference_position = np.array(target_position)
        
        # Add to reference history
        if len(self.reference_position) >= 3:  # Ensure it's a 3D position
            self.reference_history.append(self.reference_position.copy())
    
    def update_plot(self, time_step, target_position=None):
        """Update the plot with current rod positions and reference"""
        
        # Get current tip position
        current_tip = None
        if len(self.rods) > 0:
            current_tip = self.rods[-1].position_collection[:, -1]  # Last node is the tip
            self.tip_history.append(current_tip.copy())
            
            # Store initial tip position for connecting line
            if not self.trajectory_started and current_tip is not None:
                self.start_tip_position = current_tip.copy()
        
        # Update reference target if provided
        if target_position is not None:
            self.set_reference_target(target_position)
            
            # Mark trajectory as started when we first get a target
            if not self.trajectory_started:
                self.trajectory_started = True
        
        # Update rod visualizations
        for i, rod in enumerate(self.rods):
            # Get current rod position
            pos = rod.position_collection
            
            # Update XY view
            self.lines_xy[i].set_data(pos[0, :], pos[1, :])
            
            # Update XZ view  
            self.lines_xz[i].set_data(pos[0, :], pos[2, :])
            
            # Update 3D view
            self.lines_3d[i].set_data_3d(pos[0, :], pos[1, :], pos[2, :])
        
        # Update reference trajectory visualization
        if self.show_reference and self.reference_position is not None:
            # Update current target position (green point)
            self.ref_target_xy.set_data([self.reference_position[0]], [self.reference_position[1]])
            self.ref_target_xz.set_data([self.reference_position[0]], [self.reference_position[2]])
            self.ref_target_3d.set_data_3d([self.reference_position[0]], 
                                          [self.reference_position[1]], 
                                          [self.reference_position[2]])
            
            # Update reference trajectory history (green line)
            if len(self.reference_history) > 1:
                ref_hist = np.array(list(self.reference_history))
                self.ref_history_xy.set_data(ref_hist[:, 0], ref_hist[:, 1])
                self.ref_history_xz.set_data(ref_hist[:, 0], ref_hist[:, 2])
                self.ref_history_3d.set_data_3d(ref_hist[:, 0], ref_hist[:, 1], ref_hist[:, 2])
            
            # Update connecting line from initial tip position to trajectory start (dotted green)
            if (self.trajectory_started and self.start_tip_position is not None and 
                len(self.reference_history) > 0):
                first_target = list(self.reference_history)[0]
                
                # Only show connecting line if there's a significant distance
                distance = np.linalg.norm(self.start_tip_position - first_target)
                if distance > 0.01:  # 1cm threshold
                    self.connecting_line_xy.set_data(
                        [self.start_tip_position[0], first_target[0]], 
                        [self.start_tip_position[1], first_target[1]]
                    )
                    self.connecting_line_xz.set_data(
                        [self.start_tip_position[0], first_target[0]], 
                        [self.start_tip_position[2], first_target[2]]
                    )
                    self.connecting_line_3d.set_data_3d(
                        [self.start_tip_position[0], first_target[0]], 
                        [self.start_tip_position[1], first_target[1]], 
                        [self.start_tip_position[2], first_target[2]]
                    )
                else:
                    # Clear connecting line if distance is too small
                    self.connecting_line_xy.set_data([], [])
                    self.connecting_line_xz.set_data([], [])
                    self.connecting_line_3d.set_data_3d([], [], [])
            
            # Update tip trajectory history (blue line)
            if len(self.tip_history) > 1:
                tip_hist = np.array(list(self.tip_history))
                self.tip_history_xy.set_data(tip_hist[:, 0], tip_hist[:, 1])
                self.tip_history_xz.set_data(tip_hist[:, 0], tip_hist[:, 2])
                self.tip_history_3d.set_data_3d(tip_hist[:, 0], tip_hist[:, 1], tip_hist[:, 2])
            
            # Update current connection line between target and tip (thin black dotted)
            if current_tip is not None:
                self.current_connection_xy.set_data(
                    [self.reference_position[0], current_tip[0]], 
                    [self.reference_position[1], current_tip[1]]
                )
                self.current_connection_xz.set_data(
                    [self.reference_position[0], current_tip[0]], 
                    [self.reference_position[2], current_tip[2]]
                )
                self.current_connection_3d.set_data_3d(
                    [self.reference_position[0], current_tip[0]], 
                    [self.reference_position[1], current_tip[1]], 
                    [self.reference_position[2], current_tip[2]]
                )
                
                # Calculate and display tracking error
                tracking_error = np.linalg.norm(current_tip - self.reference_position)
                error_text = f'Error: {tracking_error:.4f}m'
                
                self.error_text_xy.set_text(error_text)
                self.error_text_xz.set_text(error_text)
                self.error_text_3d.set_text(error_text)
        
        # Add time annotation
        self.fig.suptitle(f'Continuum Robot Control - Time: {time_step:.3f}s', fontsize=16, fontweight='bold')
        
        # Refresh the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)  # Small pause to allow GUI to update
    
    def clear_history(self):
        """Clear trajectory history"""
        self.reference_history.clear()
        self.tip_history.clear()
        self.connecting_line_history.clear()
        self.trajectory_started = False
        self.start_tip_position = None
    
    def save_current_view(self, filename=None):
        """Save the current plot view"""
        if filename is None:
            filename = f'realtime_plot_{time.time():.0f}.png'
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved current view to {filename}")
    
    def close(self):
        """Close the plot window"""
        plt.close(self.fig)
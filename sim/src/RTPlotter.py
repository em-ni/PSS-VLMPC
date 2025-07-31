from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

class RTPlotter:
    def __init__(self, rods, xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), zlim=(-1.5, 1.5)):
        self.rods = rods
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        
        # Setup the figure and axes
        plt.ion()  # Turn on interactive mode
        self.fig = plt.figure(figsize=(15, 5))
        
        # Create subplots for different views
        self.ax_xy = self.fig.add_subplot(131)
        self.ax_xz = self.fig.add_subplot(132)
        self.ax_3d = self.fig.add_subplot(133, projection='3d')
        
        # Set limits and labels
        self.ax_xy.set_xlim(xlim)
        self.ax_xy.set_ylim(ylim)
        self.ax_xy.set_xlabel('X')
        self.ax_xy.set_ylabel('Y')
        self.ax_xy.set_title('XY View')
        self.ax_xy.set_aspect('equal')
        self.ax_xy.grid(True, alpha=0.3)
        
        self.ax_xz.set_xlim(xlim)
        self.ax_xz.set_ylim(zlim)
        self.ax_xz.set_xlabel('X')
        self.ax_xz.set_ylabel('Z')
        self.ax_xz.set_title('XZ View')
        self.ax_xz.set_aspect('equal')
        self.ax_xz.grid(True, alpha=0.3)
        
        self.ax_3d.set_xlim(xlim)
        self.ax_3d.set_ylim(ylim)
        self.ax_3d.set_zlim(zlim)
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y')
        self.ax_3d.set_zlabel('Z')
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
        
        self.ax_xy.legend()
        self.ax_xz.legend()
        self.ax_3d.legend()
        
        plt.tight_layout()
        plt.show(block=False)
        
    def update_plot(self, time_step):
        """Update the plot with current rod positions"""
        
        for i, rod in enumerate(self.rods):
            # Get current rod position
            pos = rod.position_collection
            
            # Update XY view
            self.lines_xy[i].set_data(pos[0, :], pos[1, :])
            
            # Update XZ view  
            self.lines_xz[i].set_data(pos[0, :], pos[2, :])
            
            # Update 3D view
            self.lines_3d[i].set_data_3d(pos[0, :], pos[1, :], pos[2, :])
        
        # Add time annotation
        self.fig.suptitle(f'Time: {time_step:.3f}s', fontsize=14)
        
        # Refresh the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)  # Small pause to allow GUI to update
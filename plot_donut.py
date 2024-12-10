import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_torus(R, r, u, v):
    """
    Create a torus (donut shape) with given parameters.
    
    Args:
    R (float): Major radius of the torus (distance from the center of the tube to the center of the torus)
    r (float): Minor radius of the torus (radius of the tube)
    u, v (ndarray): Meshgrid of angles parameterizing the torus
    
    Returns:
    tuple: x, y, z coordinates of the torus surface
    """
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    return x, y, z

def update(frame):
    """
    Update function for animation. Creates and plots the torus for each frame.
    
    Args:
    frame (float): Current frame number, used to calculate rotation and position
    
    Returns:
    None
    """
    ax.clear()
    
    # Create the torus
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, 2 * np.pi, 100)
    u, v = np.meshgrid(u, v)
    
    # Generate the torus coordinates
    x, y, z = create_torus(3, 1, u, v)
    
    # Adjust the position of the torus to make it "run" in a circle
    # The '2' here determines the radius of the circular path
    # The 'frame / 10' slows down the circular motion
    x += 2 * np.cos(frame / 10)
    y += 2 * np.sin(frame / 10)
    
    # Plot the surface of the torus
    ax.plot_surface(x, y, z, cmap='viridis')
    
    # Set the viewing angle, making the donut rotate as it moves
    ax.view_init(elev=20, azim=frame)
    
    # Set axis limits to keep the plot centered
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_zlim(-3, 3)
    
    # Remove axis ticks for a cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

# Create the figure and 3D axis
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Create the animation
# np.linspace(0, 2*np.pi, 120) creates 120 evenly spaced points for a full rotation
# interval=50 means each frame is displayed for 50 milliseconds
anim = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 120), interval=50, blit=False)

# Display the animation
plt.show()

# Uncomment the following lines to save the animation

# Save as GIF
# anim.save('running_donut.gif', writer='pillow', fps=30)

# Or save as MP4 (requires ffmpeg)
# anim.save('running_donut.mp4', writer='ffmpeg', fps=30)

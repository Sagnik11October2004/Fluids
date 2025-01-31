import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class FrontTracker2D:
    def __init__(self, domain_size, grid_size, ds_min=0.02, ds_max=0.05):
        """Initialize front tracker with domain parameters."""
        self.Lx, self.Ly = domain_size
        self.nx, self.ny = grid_size
        self.dx = self.Lx / (self.nx - 1)
        self.dy = self.Ly / (self.ny - 1)
        self.ds_min = ds_min
        self.ds_max = ds_max
        
        # Initialize grid
        self.x_grid = np.linspace(0, self.Lx, self.nx)
        self.y_grid = np.linspace(0, self.Ly, self.ny)
        
        # For visualization
        self.history = []
        
    def initialize_interface(self, x_points, y_points):
        """Initialize interface points."""
        self.points = np.column_stack((x_points, y_points))
        self.history = [self.points.copy()]

    def redistribute_points(self):
        """Redistribute points using cubic spline interpolation."""
        if len(self.points) < 4:
            return
            
        # Compute arc length parameterization
        diffs = np.diff(self.points, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        u = np.zeros(len(self.points))
        u[1:] = np.cumsum(segment_lengths)
        u = u/u[-1]
        
        try:
            # Fit cubic spline
            tck, _ = splprep([self.points[:, 0], self.points[:, 1]], u=u, s=0, k=3)
            
            # Generate new points based on desired spacing
            curve_length = np.sum(segment_lengths)
            n_points = max(4, int(curve_length / self.ds_max))
            u_new = np.linspace(0, 1, n_points)
            self.points = np.array(splev(u_new, tck)).T
            
        except Exception as e:
            print(f"Redistribution failed: {str(e)}")

    def get_velocity(self, x, y, velocity_field):
        """Bilinear interpolation of velocity field."""
        i = int(x / self.dx)
        j = int(y / self.dy)
        
        i = min(max(i, 0), self.nx-2)
        j = min(max(j, 0), self.ny-2)
        
        wx = (x - self.x_grid[i]) / self.dx
        wy = (y - self.y_grid[j]) / self.dy
        
        u, v = velocity_field
        u_interp = (1 - wx) * (1 - wy) * u[j,i] + \
                   wx * (1 - wy) * u[j,i+1] + \
                   (1 - wx) * wy * u[j+1,i] + \
                   wx * wy * u[j+1,i+1]
                   
        v_interp = (1 - wx) * (1 - wy) * v[j,i] + \
                   wx * (1 - wy) * v[j,i+1] + \
                   (1 - wx) * wy * v[j+1,i] + \
                   wx * wy * v[j+1,i+1]
                   
        return np.array([u_interp, v_interp])

    def rk3_step(self, dt, velocity_field):
        """Third-order Runge-Kutta time integration."""
        # First stage
        k1 = np.array([self.get_velocity(p[0], p[1], velocity_field) for p in self.points])
        p1 = self.points + dt * k1
        
        # Second stage
        k2 = np.array([self.get_velocity(p[0], p[1], velocity_field) for p in p1])
        p2 = self.points + dt * (k1/4 + k2/4)
        
        # Third stage
        k3 = np.array([self.get_velocity(p[0], p[1], velocity_field) for p in p2])
        self.points += dt * (k1/6 + k2/6 + 2*k3/3)

    def advance(self, dt, velocity_field):
        """Advance the interface one timestep."""
        self.rk3_step(dt, velocity_field)
        self.redistribute_points()
        self.history.append(self.points.copy())

class FrontVisualizer:
    def __init__(self, tracker, velocity_field):
        self.tracker = tracker
        self.velocity_field = velocity_field
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
    def setup_plot(self):
        """Setup plot with velocity field and initial interface."""
        self.ax.set_xlim(0, self.tracker.Lx)
        self.ax.set_ylim(0, self.tracker.Ly)
        
        # Plot velocity field
        X, Y = np.meshgrid(self.tracker.x_grid, self.tracker.y_grid)
        self.ax.quiver(X[::2, ::2], Y[::2, ::2], 
                      self.velocity_field[0][::2, ::2],
                      self.velocity_field[1][::2, ::2],
                      alpha=0.3)
        
        # Plot interface
        self.line, = self.ax.plot([], [], 'r-', lw=2)
        self.points, = self.ax.plot([], [], 'ko', ms=4)
        return self.line, self.points
    
    def update(self, frame):
        """Update animation frame."""
        current_points = self.tracker.history[frame]
        self.line.set_data(current_points[:, 0], current_points[:, 1])
        self.points.set_data(current_points[:, 0], current_points[:, 1])
        return self.line, self.points
    
    def animate(self, interval=50):
        """Create animation."""
        self.setup_plot()
        anim = FuncAnimation(self.fig, self.update,
                           frames=len(self.tracker.history),
                           interval=interval, blit=True)
        plt.show()
        return anim
# Setup benchmark case
def setup_single_vortex_benchmark():
    """
    Setup the single vortex benchmark case.
    The velocity field is time-dependent and reverses at t = T/2.
    """
    # Domain parameters
    domain_size = (1.0, 1.0)
    grid_size = (128, 128)  # Higher resolution for benchmark
    tracker = FrontTracker2D(domain_size, grid_size, ds_min=0.01, ds_max=0.02)
    
    # Initial circle parameters
    n_points = 100
    theta = np.linspace(0, 2*np.pi, n_points)
    r = 0.15  # Radius
    center = np.array([0.5, 0.75])  # Offset from center
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    tracker.initialize_interface(x, y)
    
    return tracker

def get_single_vortex_velocity(t, T):
    """
    Time-dependent single vortex velocity field.
    Args:
        t: Current time
        T: Period of the flow
    Returns:
        Function that computes velocity at any point (x,y)
    """
    def velocity_field(x, y):
        # Time-dependent amplitude that reverses at t = T/2
        A = np.cos(np.pi * t / T)
        
        # Stream function: sin²(πx)sin²(πy)
        psi = A * np.sin(np.pi * x)**2 * np.sin(np.pi * y)**2
        
        # Velocity components: u = -∂ψ/∂y, v = ∂ψ/∂x
        u = -2 * A * np.pi * np.sin(np.pi * x)**2 * np.sin(np.pi * y) * np.cos(np.pi * y)
        v = 2 * A * np.pi * np.sin(np.pi * y)**2 * np.sin(np.pi * x) * np.cos(np.pi * x)
        
        return np.array([u, v])
    
    return velocity_field

def run_benchmark():
    # Setup parameters
    T = 8.0  # Total period
    dt = 0.02  # Time step
    n_steps = int(T/dt)
    
    # Initialize tracker
    tracker = setup_single_vortex_benchmark()
    
    # Create grid for velocity field visualization
    x = np.linspace(0, 1, tracker.nx)
    y = np.linspace(0, 1, tracker.ny)
    X, Y = np.meshgrid(x, y)
    
    # Time stepping
    t = 0
    for step in range(n_steps):
        # Get velocity field at current time
        vel_func = get_single_vortex_velocity(t, T)
        
        # Compute velocities on grid for visualization
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        for i in range(len(x)):
            for j in range(len(y)):
                U[j,i], V[j,i] = vel_func(X[j,i], Y[j,i])
        
        # Advance interface
        tracker.advance(dt, (U, V))
        t += dt
        
        # Print progress
        if step % 50 == 0:
            print(f"Step {step}/{n_steps}, Time: {t:.2f}/{T}")
    
    # Visualize results
    visualizer = FrontVisualizer(tracker, (U, V))
    anim = visualizer.animate(interval=20)
    
    # Calculate error metrics at final time
    initial_points = tracker.history[0]
    final_points = tracker.history[-1]
    error = np.mean(np.sqrt(np.sum((final_points - initial_points)**2, axis=1)))
    print(f"\nFinal L2 error: {error:.6f}")
    
    return tracker, anim

if __name__ == "__main__":
    tracker, anim = run_benchmark()

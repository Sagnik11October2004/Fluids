import numpy as np
from scipy.fftpack import dct, idct
from scipy.sparse import diags
from scipy.sparse.linalg import bicgstab, spilu, LinearOperator
import matplotlib.pyplot as plt

def calculate_cfl_timestep(u, v, dx, dy, Re):
    """
    Calculate the maximum safe time step based on CFL condition
    
    Parameters:
    u, v : velocity fields
    dx, dy : grid spacing
    Re : Reynolds number
    
    Returns:
    Recommended time step
    """
    # Maximum velocity magnitude
    max_u = np.max(np.abs(u))
    max_v = np.max(np.abs(v))
    
    # Convective CFL condition
    cfl_conv_x = dx / (max_u + 1e-10)
    cfl_conv_y = dy / (max_v + 1e-10)
    
    # Diffusive CFL condition (based on viscosity)
    nu = 1 / Re  # Kinematic viscosity
    cfl_diff_x = 0.5 * Re * dx**2
    cfl_diff_y = 0.5 * Re * dy**2
    
    # Take the minimum to be conservative
    dt_conv = 0.5 * min(cfl_conv_x, cfl_conv_y)
    dt_diff = 0.5 * min(cfl_diff_x, cfl_diff_y)
    
    # Choose teh smaller timestep
    dt = min(dt_conv, dt_diff)
    
    print(f"CFL Analysis:")
    print(f"Max U: {max_u:.4f}, Max V: {max_v:.4f}")
    print(f"Convective CFL dt_x: {cfl_conv_x:.6f}, dt_y: {cfl_conv_y:.6f}")
    print(f"Diffusive CFL dt_x: {cfl_diff_x:.6f}, dt_y: {cfl_diff_y:.6f}")
    print(f"Recommended dt: {dt:.6f}")
    
    return dt


def solvePoissonEquation_2dDCT(b, Nx, Ny, dx, dy):
    # wavenumber
    kx = np.arange(Nx)
    ky = np.arange(Ny)
    mwx = 2 * (np.cos(np.pi * kx / Nx) - 1) / dx**2
    mwy = 2 * (np.cos(np.pi * ky / Ny) - 1) / dy**2

    # 2D DCT of b (Right hand side)
    bhat = dct(dct(b.T, norm='ortho').T, norm='ortho')

    MWX, MWY = np.meshgrid(mwx, mwy, indexing='ij')
    phat = bhat / (MWX + MWY)

    # The solution is not unique (phat[0, 0] = inf);
    denominator = MWX + MWY

    # Avoid division by zero
    denominator[0, 0] = 1  # Set the (0,0) term to 1 to avoid division by zero
    phat = bhat / denominator
    # Here we fix the mean (with kx=0, ky=0) to be 0
    phat[0, 0] = 0

    # Inverse 2D DCT
    p = idct(idct(phat.T, norm='ortho').T, norm='ortho')


    return p


    return p
def solvePoissonEquation(b, Nx, Ny, dx, dy, tol=1e-20, max_iter=10000):
    """
    Solves the Poisson equation using the Gauss-Seidel method.

    Parameters:
    b : ndarray
        Right-hand side of the Poisson equation
    Nx, Ny : int
        Number of grid points in the x and y directions
    dx, dy : float
        Grid spacing in x and y directions
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum number of iterations

    Returns:
    p : ndarray
        Solution to the Poisson equation
    """
    # Initialize the pressure field
    p = np.zeros_like(b)
    dx2, dy2 = dx**2, dy**2
    factor = 1 / (2 * (1 / dx2 + 1 / dy2))

    # Gauss-Seidel iteration
    for it in range(max_iter):
        p_old = p.copy()

        for i in range(1, Nx-1):
            for j in range(1, Ny-1):
                p[i, j] = factor * ((p[i+1, j] + p[i-1, j]) / dx2 +
                                    (p[i, j+1] + p[i, j-1]) / dy2 - b[i, j])

        # Check for convergence
        error = np.linalg.norm(p - p_old, ord=2)
        if error < tol:
            print(f"Converged in {it+1} iterations with error {error:.2e}")
            break

    return p


Re = 1000  # Reynolds number
nt = 20000  # max time steps
Lx = 1
Ly = 1  # domain size
Nx = 80
Ny = 80  # Number of grids
dt = 0.0005  # time step

# Grid size (Equispaced)
dx = Lx / Nx
dy = Ly / Ny

# Coordinate of each grid (cell center)
xce = (np.arange(1, Nx + 1) - 0.5) * dx
yce = (np.arange(1, Ny + 1) - 0.5) * dy

# Coordinate of each grid (cell corner)
xco = np.arange(0, Nx + 1) * dx
yco = np.arange(0, Ny + 1) * dy

# Data arrays
u = np.zeros((Nx + 1, Ny + 2))  # velocity in x direction (u)
v = np.zeros((Nx + 2, Ny + 1))  # velocity in y direction (v)
p = np.zeros((Nx, Ny))  # pressure (lagrange multiplier)

for ii in range(nt):
    bctop = 10  # Top lid u
    
    calculate_cfl_timestep(u, v, dx, dy, Re)
    
    if ii > 1000:
        bctop = -1
    if ii > 2000:
        bctop = 1   
    
    u[:, 0] = -u[:, 1]
    v[:, 0] = 0  # bottom
    u[:, -1] = 2 * bctop - u[:, -2]
    v[:, -1] = 0  # top
    u[0, :] = 0
    v[0, :] = -v[1, :]  # left
    u[-1, :] = 0
    v[-1, :] = -v[-2, :]  # right

    Lux = (u[:-2, 1:-1] - 2 * u[1:-1, 1:-1] + u[2:, 1:-1]) / dx**2
    Luy = (u[1:-1, :-2] - 2 * u[1:-1, 1:-1] + u[1:-1, 2:]) / dy**2
    Lvx = (v[:-2, 1:-1] - 2 * v[1:-1, 1:-1] + v[2:, 1:-1]) / dx**2
    Lvy = (v[1:-1, :-2] - 2 * v[1:-1, 1:-1] + v[1:-1, 2:]) / dy**2

    # 1. interpolate velocity at cell center/cell corner
    uce = (u[:-1, 1:-1] + u[1:, 1:-1]) / 2
    uco = (u[:, :-1] + u[:, 1:]) / 2
    vco = (v[:-1, :] + v[1:, :]) / 2
    vce = (v[1:-1, :-1] + v[1:-1, 1:]) / 2

    # 2. multiply
    uuce = uce * uce
    uvco = uco * vco
    vvce = vce * vce

    # 3-1. get derivative for u
    Nu = (uuce[1:, :] - uuce[:-1, :]) / dx
    Nu += (uvco[1:-1, 1:] - uvco[1:-1, :-1]) / dy

    # 3-2. get derivative for v
    Nv = (vvce[:, 1:] - vvce[:, :-1]) / dy
    Nv += (uvco[1:, 1:-1] - uvco[:-1, 1:-1]) / dx

    # Get intermediate velocity
    u[1:-1, 1:-1] += dt * (-Nu + (Lux + Luy) / Re)
    v[1:-1, 1:-1] += dt * (-Nv + (Lvx + Lvy) / Re)

    # velocity correction
    # RHS of pressure Poisson eq.
    b = ((u[1:, 1:-1] - u[:-1, 1:-1]) / dx +
         (v[1:-1, 1:] - v[1:-1, :-1]) / dy)

    # Solve for p (using cosine transform, faster)
    p = solvePoissonEquation_2dDCT(b, Nx, Ny, dx, dy)
    # Direct Method
    #p = solvePoissonEquation(b, Nx, Ny, dx, dy)

    # The new divergent free velocity field
    u[1:-1, 1:-1] -= (p[1:, :] - p[:-1, :]) / dx
    v[1:-1, 1:-1] -= (p[:, 1:] - p[:, :-1]) / dy

    divergence = np.abs((u[1:, 1:-1] - u[:-1, 1:-1]) / dx + (v[1:-1, 1:] - v[1:-1, :-1]) / dy)
    max_divergence = np.max(divergence)
    print(f"Time step {ii+1}, Max Divergence: {max_divergence:.2e}")


    if(ii%100==0):
        # Calculate velocity at cell centers
        uce = (u[:-1, 1:-1] + u[1:, 1:-1]) / 2
        vce = (v[1:-1, :-1] + v[1:-1, 1:]) / 2

        # Create a meshgrid for plotting
        X, Y = np.meshgrid(xce, yce, indexing='ij')

        # Plot the vector field and pressure field side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the velocity vector field
        ax1.quiver(X, Y, uce, vce)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('Velocity Vector Field')

        # Plot the pressure field
        pressure_plot = ax2.imshow(p.T, origin='lower', extent=[xce[0], xce[-1], yce[0], yce[-1]], aspect='auto',cmap='plasma')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('Pressure Field')
        fig.colorbar(pressure_plot, ax=ax2, orientation='vertical')
        plt.show()

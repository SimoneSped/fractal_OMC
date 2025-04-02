import numpy as np

# Function to generate a Gaussian with potential cut-offs
def generate_2d_gaussian(shape, center, sigma):
    """
    Generate a Gaussian distribution that may cut off at the edges of the data.
    
    Parameters:
    - shape: Tuple (nx, ny), the size of the 2D grid.
    - center: Tuple (x0, y0), the center of the Gaussian.
    - sigma: Standard deviation of the Gaussian.
    
    Returns:
    - gaussian: 2D array with the Gaussian distribution.
    """
    x = np.arange(0, shape[0])
    y = np.arange(0, shape[1])
    xx, yy = np.meshgrid(x, y)
    
    gaussian = np.exp(-(((xx - center[0])**2 + (yy - center[1])**2) / (2 * sigma**2)))
    return gaussian